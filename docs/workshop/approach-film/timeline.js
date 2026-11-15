const byId = (id) => document.getElementById(id);
const parts = Object.fromEntries(
  Array.from(document.querySelectorAll("[id]")).map((element) => [element.id, element]),
);

const routes = [
  { element: byId("frames-route"), cue: "frames", offset: 0, rank: 3 },
  { element: byId("whisper-route"), cue: "qwen_video", offset: 0.35, rank: 1 },
  { element: byId("video-route"), cue: "qwen_video", offset: 1.65, rank: 2 },
  { element: byId("gemini-route"), cue: "gemini", offset: 0.2, rank: 0 },
];

const clamp = (value) => Math.min(1, Math.max(0, value));
const mix = (start, end, amount) => start + (end - start) * amount;

function ease(time, start, end) {
  const unit = clamp((time - start) / (end - start));
  return unit * unit * (3 - 2 * unit);
}

function reveal(element, amount, shift = 0) {
  element.style.opacity = amount;
  element.style.transform = shift ? `translateY(${shift * (1 - amount)}px)` : "";
}

function fillBars(id, time, start) {
  parts[id].querySelectorAll(".fill").forEach((fill, index) => {
    const begins = start + index * 0.9;
    const progress = ease(time, begins, begins + 0.85);
    fill.style.width = `${Number(fill.dataset.portion) * progress * 100}%`;
  });
}

window.setFilmTimeline = (timeline) => {
  const at = (cue) => timeline.cues[cue].start;
  const frames = at("frames");
  const assemble = at("assemble");
  const wording = at("wording");
  const matching = at("match");
  const direct = at("direct");
  const test = at("test");
  const meteor = at("meteor");
  const accuracy = at("accuracy");
  const mcRank = at("mc_rank");
  const end = at("end");

  window.drawFilm = (time) => {
    const opening = 1 - ease(time, frames - 1.22, frames - 0.77);
    const generation = ease(time, frames - 0.45, frames + 0.24)
      * (1 - ease(time, matching - 0.5, matching));
    const choice = ease(time, matching - 0.12, matching + 0.55)
      * (1 - ease(time, test - 0.58, test));
    const results = ease(time, test - 0.1, test + 0.58);

    reveal(parts.opening, opening, -12);
    reveal(parts.tasks, ease(time, at("tasks") + 0.05, at("tasks") + 0.85), 10);

    const dock = ease(time, frames - 0.86, frames + 0.14);
    const x = mix(868, 82, dock);
    const y = mix(192, 250, dock);
    parts.clip.style.transform = `translate(${x}px,${y}px) scale(${mix(1, 0.76, dock)})`;
    parts.clip.style.opacity = 1 - ease(time, test - 0.5, test + 0.1);

    reveal(parts["generation-title"], generation, 8);
    reveal(parts.routes, generation);
    reveal(parts.branch, generation * ease(time, frames + 0.12, frames + 0.74));

    const sort = ease(time, assemble + 2.02, assemble + 3.4);
    const score = ease(time, wording + 2.1, wording + 2.75);
    routes.forEach(({ element, cue, offset, rank }, index) => {
      const arrives = at(cue) + offset;
      const moveIn = ease(time, arrives, arrives + 0.55);
      const position = mix(index, rank, sort);
      element.style.opacity = moveIn;
      element.style.transform = `translate(${16 * (1 - moveIn)}px,${position * 113}px)`;
      element.style.zIndex = index === 3 ? "2" : "1";
      element.classList.toggle("preferred", rank === 0 && sort > 0.88);
      reveal(element.querySelector(".reply"), ease(time, assemble + 2.8, assemble + 3.55));
      const exampleScore = element.querySelector(".meter-score");
      if (exampleScore) reveal(exampleScore, score);
    });
    reveal(parts["wording-note"], generation * score, 12);

    reveal(parts["choice-title"], choice, 8);
    reveal(parts.choice, choice);
    reveal(parts["matching-method"], ease(time, matching + 0.6, matching + 1.25), 12);
    reveal(parts["match-pick"], ease(time, matching + 2.05, matching + 2.52));
    reveal(parts["ranking-method"], ease(time, direct - 0.1, direct + 0.55), 12);
    reveal(parts["video-pick"], ease(time, direct + 1.1, direct + 1.72));
    reveal(parts["choice-key"], choice * ease(time, direct + 2.8, direct + 3.3), 12);

    reveal(parts["scores-title"], results, 10);
    reveal(parts.scores, results);
    const revealBoth = ease(time, accuracy - 0.32, accuracy + 0.66);
    parts["ag-score"].style.opacity = ease(time, test + 1.0, test + 1.75);
    parts["ag-score"].style.transform = `translateX(${mix(362, 0, revealBoth)}px)`;
    reveal(parts["ag-rank"], ease(time, meteor + 4.0, meteor + 4.7));
    reveal(parts["mc-score"], revealBoth, 16);
    reveal(parts["mc-standing"], ease(time, mcRank + 0.7, mcRank + 1.2));
    fillBars("ag-score", time, meteor + 0.4);
    fillBars("mc-score", time, accuracy + 1.3);

    reveal(parts.closing, results);
    reveal(parts.gain, ease(time, mcRank + 3.5, mcRank + 4.1), 10);
    reveal(parts.paper, ease(time, end, end + 0.6), 10);
    parts.progress.style.transform = `scaleX(${clamp(time / timeline.duration)})`;
  };

  window.drawFilm(0);
};
