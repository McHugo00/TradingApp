/**
 * Validate predictions by comparing them against actual close values.
 * @param {import('mongodb').Db} db
 * @param {{ onlyMissing?: boolean, limit?: number }} [options]
 */
export async function predictionValidation(db, options = {}) {
  const predictionsCol = db.collection('prediction');

  const query = {};
  if (options.onlyMissing) {
    query.actual = { $exists: false };
  }

  const limit = typeof options.limit === 'number' && options.limit > 0 ? options.limit : 500;

  const cursor = predictionsCol.find(query).limit(limit);

  const updates = [];
  const now = new Date();

  const toNumber = (value) => {
    if (value === null || value === undefined) return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };

  while (await cursor.hasNext()) {
    const doc = await cursor.next();
    if (!doc) continue;

    const targetCollectionName =
      typeof doc.collection === 'string' ? doc.collection.trim() : '';

    if (!targetCollectionName) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'missing_collection_name',
              validated_at: now
            }
          }
        }
      });
      continue;
    }

    const expectedTime = doc.expectedtime;
    if (!expectedTime) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'missing_expected_time',
              validated_at: now
            }
          }
        }
      });
      continue;
    }

    let actualRecord = null;
    try {
      actualRecord = await db
        .collection(targetCollectionName)
        .findOne({ expectedtime: expectedTime });
    } catch (error) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'collection_lookup_failed',
              validation_error: error && error.message ? error.message : String(error),
              validated_at: now
            }
          }
        }
      });
      continue;
    }

    const actualRaw =
      actualRecord && actualRecord.closeprice !== undefined
        ? actualRecord.closeprice
        : actualRecord && actualRecord.close !== undefined
        ? actualRecord.close
        : null;

    const actual = toNumber(actualRaw);

    if (actual === null) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'missing_actual_value',
              validated_at: now
            }
          }
        }
      });
      continue;
    }

    const prediction = toNumber(doc.prediction);
    const inputClose = toNumber(doc.input_ClosePrice);

    const valueError =
      prediction !== null ? actual - prediction : null;
    const absError =
      valueError !== null ? Math.abs(valueError) : null;
    const pctError =
      valueError !== null && actual !== 0 ? Math.abs(valueError / actual) : null;

    const actualDir =
      inputClose !== null ? Math.sign(actual - inputClose) : null;
    const predictedDir =
      prediction !== null && inputClose !== null ? Math.sign(prediction - inputClose) : null;

    const directionCorrect =
      actualDir !== null && predictedDir !== null
        ? actualDir === predictedDir
        : null;

    updates.push({
      updateOne: {
        filter: { _id: doc._id },
        update: {
          $set: {
            actual,
            actual_source_collection: targetCollectionName,
            actual_source_id: actualRecord && actualRecord._id ? actualRecord._id : null,
            absolute_error: absError,
            percent_error: pctError,
            direction_correct: directionCorrect,
            validation_status: 'validated',
            validated_at: now
          },
          $unset: {
            validation_error: ''
          }
        }
      }
    });
  }

  if (updates.length) {
    await predictionsCol.bulkWrite(updates, { ordered: false });
  }

  return { processed: updates.length };
}
