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
  query.validation_status = { $ne: 'validated' };

  const limit = typeof options.limit === 'number' && options.limit > 0 ? options.limit : 500;

  const cursor = predictionsCol.find(query).limit(limit);

  const updates = [];
  const now = new Date();

  const timeFields = ['expectedtime', 'expected_time', 't', 'timestamp', 'Timestamp'];

  const normalizeName = (candidate) => {
    if (candidate === null || candidate === undefined) return '';
    return candidate.toString().trim().replace(/[^a-z0-9]/gi, '').toLowerCase();
  };

  const aliasGroups = {
    actual: ['actual', 'actualvalue', 'actual_price', 'actualprice'],
    closeprice: ['closeprice', 'close_price', 'closePrice', 'ClosePrice', 'closingprice', 'closing_price'],
    close: ['close', 'Close', 'closing', 'closingPrice', 'closing_price', 'closevalue', 'close_value'],
    lastprice: ['lastprice', 'last_price', 'lastPrice', 'LastPrice', 'last_trade_price', 'lasttradeprice'],
    marketclose: ['marketclose', 'market_close', 'regularmarketclose', 'regular_market_close'],
    mark: ['mark', 'markprice', 'mark_price', 'MarkPrice'],
    c: ['c', 'C'],
    price: [
      'price',
      'Price',
      'tradeprice',
      'trade_price',
      'marketprice',
      'market_price',
      'regularmarketprice',
      'regularMarketPrice',
      'avgprice',
      'averageprice',
      'AveragePrice'
    ]
  };

  const canonicalPriority = Object.keys(aliasGroups);

  const aliasLookup = new Map();
  for (const [canonical, aliases] of Object.entries(aliasGroups)) {
    aliasLookup.set(normalizeName(canonical), canonical);
    for (const alias of aliases) {
      aliasLookup.set(normalizeName(alias), canonical);
    }
  }

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

  const resolveNumeric = (value, depth = 0) => {
    if (depth > 5) return null;

    const direct = toNumber(value);
    if (direct !== null) return direct;

    if (Array.isArray(value)) {
      for (const item of value) {
        const num = resolveNumeric(item, depth + 1);
        if (num !== null) return num;
      }
      return null;
    }

    if (value && typeof value === 'object') {
      for (const val of Object.values(value)) {
        const num = resolveNumeric(val, depth + 1);
        if (num !== null) return num;
      }
    }

    return null;
  };

  const isPrimitive = (value) =>
    value === null || (typeof value !== 'object' && typeof value !== 'function');

  const extractActualValue = (record) => {
    if (!record || typeof record !== 'object') return null;

    const visited = new Set();
    const queue = [record];
    const candidates = new Map();
    const MAX_VISITS = 500;
    let visitedCount = 0;

    while (queue.length && visitedCount < MAX_VISITS) {
      const current = queue.shift();
      if (!current || typeof current !== 'object') continue;
      if (visited.has(current)) continue;
      visited.add(current);
      visitedCount += 1;

      const isArray = Array.isArray(current);
      const entries = isArray ? current.entries() : Object.entries(current);

      for (const [rawKey, value] of entries) {
        const key = isArray ? String(rawKey) : rawKey;
        const normalized = normalizeName(key);
        const canonical = aliasLookup.get(normalized);

        if (canonical) {
          const isCurrentPrimitive = isPrimitive(value);
          const existing = candidates.get(canonical);
          if (!existing || (isCurrentPrimitive && !isPrimitive(existing.value))) {
            candidates.set(canonical, { value });
          }
        }

        if (value && typeof value === 'object' && !visited.has(value)) {
          queue.push(value);
        }
      }
    }

    for (const canonical of canonicalPriority) {
      const candidate = candidates.get(canonical);
      if (!candidate) continue;
      const resolved = resolveNumeric(candidate.value);
      if (resolved !== null) return resolved;
    }

    for (const candidate of candidates.values()) {
      const resolved = resolveNumeric(candidate.value);
      if (resolved !== null) return resolved;
    }

    return null;
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

    let expectedDate = null;
    if (expectedTime instanceof Date && !Number.isNaN(expectedTime.getTime())) {
      expectedDate = new Date(expectedTime.getTime());
    } else {
      const parsedExpected = new Date(expectedTime);
      if (!Number.isNaN(parsedExpected.getTime())) expectedDate = parsedExpected;
    }

    const expectedIso = expectedDate ? expectedDate.toISOString() : null;

    const symbol =
      typeof doc.symbol === 'string' ? doc.symbol.trim().toUpperCase() : null;

    if (!symbol) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'missing_symbol',
              validation_error: 'Prediction document missing symbol for actual lookup',
              validated_at: now
            }
          }
        }
      });
      continue;
    }

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
    if (expectedDate) addTimeCandidate(expectedDate);

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
    let collection;
    try {
      collection = db.collection(targetCollectionName);
      const query = { symbol, $or: orClauses };
      actualRecord = await collection.findOne(query, { sort: { t: -1 } });
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

    if (!actualRecord && collection) {
      const rangeClauses = [];

      if (expectedDate) {
        for (const field of timeFields) {
          rangeClauses.push({ [field]: { $gte: expectedDate } });
        }
      }

      if (expectedIso) {
        for (const field of timeFields) {
          rangeClauses.push({ [field]: { $gte: expectedIso } });
        }
      }

      if (rangeClauses.length) {
        try {
          actualRecord = await collection.findOne(
            { symbol, $or: rangeClauses },
            { sort: { t: 1 } }
          );
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
      }
    }

    if (!actualRecord) {
      updates.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              validation_status: 'actual_not_found',
              validation_error: `No matching actual record found in ${targetCollectionName} for symbol ${symbol} at ${expectedTime} (searched for later records as well)`,
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
