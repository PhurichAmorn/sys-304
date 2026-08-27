const test = require("node:test");
const assert = require("node:assert");
const { predictTweet, formatResult } = require("../app.js");

function mockFetch(status, body) {
  return async () => ({
    ok: status === 200,
    status,
    json: async () => body,
  });
}

// Integration test 1: happy path end-to-end through predictTweet -> formatResult.
test("predictTweet resolves and formats a disaster prediction", async () => {
  const fetchImpl = mockFetch(200, { prediction: 1, probability: 0.87 });
  const data = await predictTweet("http://api", "wildfire spreading fast", fetchImpl);
  const { label, cls, probability } = formatResult(data);
  assert.strictEqual(label, "DISASTER");
  assert.strictEqual(cls, "disaster");
  assert.strictEqual(probability, "0.870");
});

// Integration test 2: API error status is surfaced as a thrown error.
test("predictTweet throws when the API responds with a non-200 status", async () => {
  const fetchImpl = mockFetch(500, {});
  await assert.rejects(() => predictTweet("http://api", "", fetchImpl), /API returned 500/);
});

test("formatResult labels non-disaster predictions correctly", () => {
  const { label, cls } = formatResult({ prediction: 0, probability: 0.12 });
  assert.strictEqual(label, "NOT a disaster");
  assert.strictEqual(cls, "not-disaster");
});
