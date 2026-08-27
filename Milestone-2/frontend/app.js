// Pure logic, kept separate from the DOM so it can be unit-tested with node:test.
async function predictTweet(apiUrl, text, fetchImpl) {
  const res = await fetchImpl(`${apiUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    throw new Error(`API returned ${res.status}`);
  }
  return res.json();
}

function formatResult(data) {
  const label = data.prediction === 1 ? "DISASTER" : "NOT a disaster";
  const cls = data.prediction === 1 ? "disaster" : "not-disaster";
  return { label, cls, probability: data.probability.toFixed(3) };
}

if (typeof module !== "undefined") {
  module.exports = { predictTweet, formatResult };
}
