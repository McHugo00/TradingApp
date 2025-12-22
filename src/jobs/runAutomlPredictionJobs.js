import { spawn } from 'node:child_process';
import { connectDb, getDb, closeDb } from '../db.js';

const COLLECTION_TARGETS = [
  { field: 'market_snapshots_1m', collection: 'market_snapshots_1m' },
  { field: '1h_bars', collection: '1h_bars' },
  { field: '15m_bars', collection: '15m_bars' },
  { field: '1d_bars', collection: '1d_bars' },
  { field: '1m_bars', collection: '1m_bars' }
];

function hasTrainedDate(value) {
  if (!value) return false;
  const dateValue = value instanceof Date ? value : new Date(value);
  return !Number.isNaN(dateValue.getTime());
}

function normalizeSymbol(raw) {
  if (!raw) return '';
  return raw.toString().trim().toUpperCase();
}

function runAutomlProcess(pythonBin, collection, symbol) {
  return new Promise((resolve) => {
    const args = [
      '-m',
      'ml.run_automl',
      '--collection',
      collection,
      '--symbol',
      symbol,
      '--predict-next',
      '--predict-only'
    ];

    const child = spawn(pythonBin, args, { stdio: 'inherit' });

    child.on('error', (error) => {
      resolve({ success: false, error });
    });

    child.on('exit', (code, signal) => {
      if (code === 0) {
        resolve({ success: true });
        return;
      }
      const message =
        code !== null
          ? `python exited with code ${code}`
          : `python terminated by signal ${signal || 'unknown'}`;
      resolve({ success: false, error: new Error(message) });
    });
  });
}

export async function runAutomlPredictionJobs(options = {}) {
  const pythonBin = options.pythonBin || process.env.PYTHON_BIN || 'python';
  const reuseConnection = options.reuseConnection === true;

  const normalizedFields = Array.isArray(options.fields)
    ? options.fields
        .map((field) => (field === undefined || field === null ? '' : field.toString().trim()))
        .filter(Boolean)
    : [];

  if (normalizedFields.length) {
    const knownTargets = new Set(
      COLLECTION_TARGETS.flatMap((target) => [target.field, target.collection])
    );
    const unknown = normalizedFields.filter((field) => !knownTargets.has(field));
    if (unknown.length) {
      console.warn(`[automl] Ignoring unknown collection field(s): ${unknown.join(', ')}`);
    }
  }

  const targets =
    normalizedFields.length > 0
      ? COLLECTION_TARGETS.filter(
          (target) =>
            normalizedFields.includes(target.field) || normalizedFields.includes(target.collection)
        )
      : COLLECTION_TARGETS;

  if (!targets.length) {
    return {
      symbolsEvaluated: 0,
      jobsPlanned: 0,
      jobsSucceeded: 0,
      jobsFailed: 0,
      targetFields: []
    };
  }

  if (!reuseConnection) {
    await connectDb();
  }
  try {
    const db = await getDb();
    const projection = { symbol: 1 };
    for (const target of targets) {
      projection[target.field] = 1;
    }

    const cursor = db.collection('listeningsymbols').find({}, { projection });

    const jobs = [];
    let symbolsEvaluated = 0;

    while (await cursor.hasNext()) {
      const doc = await cursor.next();
      if (!doc) continue;
      const symbol = normalizeSymbol(doc.symbol);
      if (!symbol) continue;

      symbolsEvaluated += 1;

      for (const target of targets) {
        if (hasTrainedDate(doc[target.field])) {
          jobs.push({
            symbol,
            collection: target.collection,
            field: target.field
          });
        }
      }
    }

    await cursor.close();

    let jobsSucceeded = 0;
    let jobsFailed = 0;

    for (const job of jobs) {
      console.log(
        `[automl] Running prediction for ${job.symbol} via collection ${job.collection} (trigger: ${job.field})`
      );
      const result = await runAutomlProcess(pythonBin, job.collection, job.symbol);
      if (result.success) {
        jobsSucceeded += 1;
        console.log(
          `[automl] Completed prediction for ${job.symbol} (${job.collection})`
        );
      } else {
        jobsFailed += 1;
        console.warn(
          `[automl] Prediction failed for ${job.symbol} (${job.collection}):`,
          result.error ? result.error.message : 'unknown error'
        );
      }
    }

    return {
      symbolsEvaluated,
      jobsPlanned: jobs.length,
      jobsSucceeded,
      jobsFailed,
      targetFields: targets.map((t) => t.field)
    };
  } finally {
    if (!reuseConnection) {
      try {
        await closeDb();
      } catch (err) {
        console.warn('[automl] Failed to close DB connection:', err);
      }
    }
  }
}

async function cli() {
  try {
    const summary = await runAutomlPredictionJobs();
    console.log(
      `[automl] Finished. symbols=${summary.symbolsEvaluated} jobs=${summary.jobsPlanned} success=${summary.jobsSucceeded} failed=${summary.jobsFailed}`
    );
    process.exit(summary.jobsFailed > 0 ? 1 : 0);
  } catch (err) {
    console.error('[automl] Fatal error:', err);
    process.exit(1);
  }
}

const isCli = (() => {
  if (!process.argv[1]) return false;
  try {
    return import.meta.url === new URL(`file://${process.argv[1]}`).href;
  } catch {
    return false;
  }
})();

if (isCli) {
  cli();
}
