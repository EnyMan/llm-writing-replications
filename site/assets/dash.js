/* Minimal stand-in for the dashboard runtime the report fragments were written for, so they run as static pages.
   The page lists its datasets in <body data-datasets="a b">; each loads from data/<id>.json (rows as an array).
     dash.data(id)        -> {status: "loading" | "ok" | "error", data, message}
     dash.calc(id, spec)  derived value from datasets: spec.inputs (ids), spec.fn(...rows) -> rows or a value
     dash.draw(name, fn)  fn(el) for every element with data-draw="name", after data loads and on resize/theme change
     dash.colors          series colors from the CSS tokens (--series-1..3), light or dark */
(function () {
  const sets = {}, calcs = {}, draws = [];
  let memo = {};
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  window.dash = {
    get colors() { return [css("--series-1"), css("--series-2"), css("--series-3")]; },
    data(id) {
      if (calcs[id]) return (memo[id] ??= runCalc(id));
      return sets[id] || { status: "loading", data: [] };
    },
    calc(id, spec) { calcs[id] = spec; },
    draw(name, fn) { draws.push([name, fn]); },
  };
  function runCalc(id) {
    const ins = calcs[id].inputs.map((i) => window.dash.data(i));
    const bad = ins.find((x) => x.status !== "ok");
    if (bad) return { status: bad.status, data: null, message: bad.message };
    try { return { status: "ok", data: calcs[id].fn(...ins.map((x) => x.data)) }; }
    catch (e) { return { status: "error", data: null, message: String(e) }; }
  }
  function render() {
    memo = {};
    for (const [name, fn] of draws) {
      document.querySelectorAll(`[data-draw="${name}"]`).forEach((el) => {
        try { fn(el); } catch (e) { console.error(`draw ${name}:`, e); }
      });
    }
  }
  async function load() {
    const ids = (document.body.dataset.datasets || "").split(/\s+/).filter(Boolean);
    render();  // placeholders while loading
    await Promise.all(ids.map(async (id) => {
      try {
        const r = await fetch(`data/${id}.json`);
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        sets[id] = { status: "ok", data: await r.json() };
      } catch (e) { sets[id] = { status: "error", data: [], message: String(e) }; }
    }));
    render();
  }
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(render, 150); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  document.addEventListener("DOMContentLoaded", load);
})();
