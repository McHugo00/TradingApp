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

  const timeFields = ['expectedtime', 'expected_time', 't', 'timestamp', 'Timestamp'];
  const actualValueKeys = [
    'actual',
    'close',
    'closeprice',
    'closePrice',
    'c',
    'price',
    'last_price',
    'lastPrice',
    'closing_price'
  ];

  const extractActualValue = (record) => {
    if (!record || typeof record !== 'object') return null;
    const visited = new Set();
    const queue = [record];
    const MAX_NODES = 50;
    let processed = 0;

    while (queue.length && processed < MAX_NODES) {
      const current = queue.shift();
      if (!current || typeof current !== 'object') continue;
      if (visited.has(current)) continue;
      visited.add(current);
      processed += 1;

      for (const key of actualValueKeys) {
        if (Object.prototype.hasOwnProperty.call(current, key)) {
          const value = current[key];
          if (value !== undefined && value !== null) {
            return value;
          }
        }
      }

      for (const value of Object.values(current)) {
        if (value && typeof value === 'object' && !visited.has(value)) {
          queue.push(value);
        }
      }
    }

    return null;
  };

  const normaliseTimestamp = (value) => {
    if (!value) return null;
    if (value instanceof Date) {
      const ms = value.getTime();
      return Number.isNaN(ms) ? null : value.toISOString();
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
      const fromNumber = new Date(value);
      return Number.isNaN(fromNumber.getTime()) ? null : fromNumber.toISOString();
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      return trimmed || null;
    }
    return null;
  };

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

    const symbol =
      typeof doc.symbol === 'string' ? doc.symbol.trim().toUpperCase() : null;

    const timeStringCandidates = new Set();
    const timeDateCandidates = [];
    const seenDateMs = new Set();

    const addDateCandidate = (dateValue) => {
      if (!(dateValue instanceof Date)) return;
      const ms = dateValue.getTime();
      if (Number.isNaN(ms) || seenDateMs.has(ms)) return;
      seenDateMs.add(ms);
      timeDateCandidates.push(dateValue);
      timeStringCandidates.add(dateValue.toISOString());
    };

    const addTimeCandidate = (value) => {
      if (value === null || value === undefined) return;
      if (value instanceof Date) {
        addDateCandidate(value);
        return;
      }
      if (typeof value === 'number' && Number.isFinite(value)) {
        const dateFromNumber = new Date(value);
        if (!Number.isNaN(dateFromNumber.getTime())) {
          addDateCandidate(dateFromNumber);
        }
        return;
      }
      const str = String(value).trim();
      if (!str) return;
      timeStringCandidates.add(str);
      const parsed = new Date(str);
      if (!Number.isNaN(parsed.getTime())) {
        addDateCandidate(parsed);
      }
    };

    addTimeCandidate(expectedTime);

    const stringCandidates = Array.from(timeStringCandidates);
    const dateCandidates = timeDateCandidates.slice();

    const orClauses = [];

    if (stringCandidates.length) {
      for (const field of timeFields) {
        orClauses.push({ [field]: { $in: stringCandidates } });
      }
    }

    if (dateCandidates.length) {
      for (const field of timeFields) {
        orClauses.push({ [field]: { $in: dateCandidates } });
      }
    }

    if (!orClauses.length) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'actual_not_found',
              validation_error: 'Unable to derive time filters for actual lookup',
              validated_at: now,
              actual_source_collection: targetCollectionName
            }
          }
        }
      });
      continue;
    }

    let actualRecord = null;
    try {
      const collection = db.collection(targetCollectionName);
      const baseFilter = symbol ? { symbol } : {};
      const query = { ...baseFilter, $or: orClauses };
      actualRecord = await collection.findOne(query, { sort: { t: -1 } });

      if (!actualRecord && symbol) {
        actualRecord = await collection.findOne({ $or: orClauses }, { sort: { t: -1 } });
      }
    } catch (error) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'collection_lookup_failed',
              validation_error: error && error.message ? error.message : String(error),
              validated_at: now,
              actual_source_collection: targetCollectionName
            }
          }
        }
      });
      continue;
    }

    if (!actualRecord) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'actual_not_found',
              validation_error: `No matching actual record found in ${targetCollectionName}`,
              validated_at: now,
              actual_source_collection: targetCollectionName
            }
          }
        }
      });
      continue;
    }

    const actualRaw = extractActualValue(actualRecord);
    const actual = toNumber(actualRaw);

    const actualSourceId = actualRecord && actualRecord._id ? actualRecord._id : null;
    const actualTimestamp = normaliseTimestamp(
      actualRecord &&
        (actualRecord.t ??
          actualRecord.expectedtime ??
          actualRecord.expected_time ??
          actualRecord.timestamp ??
          actualRecord.Timestamp)
    );

    if (actual === null) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'missing_actual_value',
              validation_error: 'Actual record missing a numeric close value',
              validated_at: now,
              actual_source_collection: targetCollectionName,
              actual_source_id: actualSourceId,
              actual_timestamp: actualTimestamp
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
            actual_source_id: actualSourceId,
            actual_timestamp: actualTimestamp,
            value_error: valueError,
            absolute_error: absError,
            percent_error: pctError,
            direction_actual: actualDir,
            direction_predicted: predictedDir,
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
