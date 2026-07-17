#!/usr/bin/env node

import { randomUUID } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

// Canonical portable-artifact delivery with one compatibility correction for
// the packaged reader's sticky header. The upstream runtime sizes that header
// to 100vw, which includes classic scrollbar width and causes a small page-level
// horizontal overflow. The report payload, semantic fallback, validation,
// static-chart extraction, and browser verification remain on the shared path.
import {
  buildPortableArtifact,
  readPackagedReaderRuntime,
} from "/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/build_portable_artifact.mjs";
import { deliverPortableArtifact } from "/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs";
import { extractPortableChartSvgs } from "/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/extract_portable_chart_svgs.mjs";
import {
  chromiumDumpArguments,
  spawnChromiumDump,
} from "/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/portable_browser_cli.mjs";
import { verifyPortableArtifactStructure } from "/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/verify_portable_artifact.mjs";

const browserExecutable = "/usr/bin/google-chrome";

const originalCss = `  width: 100vw;
  height: 48px;
  min-height: 48px;
  margin-right: calc(50% - 50vw);
  margin-left: calc(50% - 50vw);`;
const correctedCss = `  width: 100%;
  height: 48px;
  min-height: 48px;
  margin-right: 0;
  margin-left: 0;`;

const packagedRuntime = readPackagedReaderRuntime().html;
const occurrences = packagedRuntime.split(originalCss).length - 1;
if (occurrences !== 1) {
  throw new Error(`Expected one packaged header rule to patch; found ${occurrences}.`);
}
const runtimeHtml = packagedRuntime.replace(originalCss, correctedCss);

function build(input, options = {}) {
  return buildPortableArtifact(input, { ...options, runtimeHtml });
}

function runChromeDump(options) {
  const browserArguments = options.arguments.map((argument) =>
    argument === "--window-size=1200,900" ? "--window-size=1200,1043" : argument,
  );
  return spawnChromiumDump({ ...options, arguments: browserArguments });
}

function extract(options = {}) {
  return extractPortableChartSvgs({
    ...options,
    browserExecutable,
    runDump: runChromeDump,
  });
}

function encodeJson(value) {
  return Buffer.from(JSON.stringify(value), "utf8").toString("base64");
}

// Runs inside one report iframe. Keeping desktop and mobile in separate browser
// invocations avoids full Chrome's dependency-free dump-DOM race with two large
// srcdoc frames while retaining the shared verifier's substantive checks.
function portableReportProbe(encodedConfig) {
  const decode = (value) => JSON.parse(new TextDecoder().decode(
    Uint8Array.from(atob(value), (character) => character.charCodeAt(0)),
  ));
  const config = decode(encodedConfig);
  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const visible = (node) => {
    if (!(node instanceof Element)) return false;
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return !node.hidden && style.display !== "none" && style.visibility !== "hidden" &&
      rect.width > 0 && rect.height > 0;
  };
  const waitUntil = async (predicate, timeoutMs) => {
    const deadline = performance.now() + timeoutMs;
    do {
      const result = predicate();
      if (result) return result;
      await delay(16);
    } while (performance.now() < deadline);
    return null;
  };
  const fail = (code, message, details = {}) => {
    const error = new Error(message);
    error.code = code;
    error.details = details;
    throw error;
  };
  const check = (condition, code, message, details = {}) => {
    if (!condition) fail(code, message, details);
  };
  const counts = () => ({
    blocks: document.querySelectorAll("[data-analytics-layout-item]").length,
    charts: document.querySelectorAll(".chart-frame").length,
    html: document.querySelectorAll("iframe.report-html-frame").length,
    metrics: document.querySelectorAll(".report-metric-card").length,
    tables: document.querySelectorAll(".table-card").length,
  });
  const sameCounts = (actual, expected) =>
    ["blocks", "charts", "html", "metrics", "tables"]
      .every((key) => actual[key] === expected[key]);
  const send = (result) => parent.postMessage({
    channel: config.channel,
    result: { ...result, viewport: config.viewport },
  }, "*");

  const run = async () => {
    try {
      const terminal = new Set(["ready", "failed", "missing-runtime", "unsupported"]);
      const readerState = await waitUntil(() => {
        const state = document.documentElement.dataset.dataAnalyticsPortableReader ?? "";
        return terminal.has(state) ? state : null;
      }, config.readyTimeoutMs);
      check(readerState, "reader_timeout",
        `Portable reader did not reach a terminal state within ${config.readyTimeoutMs}ms.`,
        { state: document.documentElement.dataset.dataAnalyticsPortableReader ?? "unset" });
      check(readerState === "ready", `reader_${readerState}`,
        `Portable reader entered terminal failure state: ${readerState}.`);
      check(matchMedia("(prefers-reduced-motion: reduce)").matches,
        "browser_configuration", "Reduced-motion mode was not applied.");
      check(matchMedia("(prefers-color-scheme: light)").matches,
        "browser_configuration", "Deterministic light theme was not applied.");
      check(innerWidth === config.viewport.width && innerHeight === config.viewport.height,
        "viewport_mismatch", "Report viewport does not match the requested size.",
        { actual: { width: innerWidth, height: innerHeight }, expected: config.viewport });
      await delay(64);

      const root = document.querySelector("#data-analytics-portable-reader");
      check(visible(root), "reader_not_visible", "Portable reader root is not visible.");
      check(root.querySelector("h1")?.textContent?.trim() === config.title.trim(),
        "title_mismatch", "Rendered title does not match the artifact title.");
      const actualCounts = counts();
      check(sameCounts(actualCounts, config.expectedCounts), "count_mismatch",
        "Rendered block counts do not match the embedded artifact.",
        { actual: actualCounts, expected: config.expectedCounts });
      const invalidGeometry = [
        "[data-analytics-layout-item]", ".chart-frame", ".table-card", ".report-metric-card",
      ].flatMap((selector) => Array.from(root.querySelectorAll(selector))
        .filter((node) => !visible(node)).map((node) => node.id || node.className || selector));
      check(invalidGeometry.length === 0, "content_not_visible",
        "One or more report elements have zero width or height.",
        { invalid: invalidGeometry.slice(0, 20) });
      const fallback = document.querySelector("[data-portable-fallback]");
      check(fallback && !visible(fallback), "fallback_visible",
        "Semantic fallback is missing or still visible after reader startup.");
      await waitUntil(() => document.documentElement.scrollWidth <=
        document.documentElement.clientWidth + 1, 1_000);
      check(document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
        "horizontal_overflow", `Report overflows horizontally at ${config.viewport.width}px.`, {
          clientWidth: document.documentElement.clientWidth,
          scrollWidth: document.documentElement.scrollWidth,
        });

      let sourceInteraction = "not_applicable";
      if (config.checkSource) {
        const button = root.querySelector(
          'button[data-artifact-action="open-options"][data-artifact-has-source="true"]',
        );
        check(button instanceof HTMLButtonElement, "source_control_missing",
          "No semantic source-options button was rendered.");
        button.click();
        const menu = await waitUntil(() => Array.from(document.querySelectorAll('[role="menu"]'))
          .find(visible), config.actionTimeoutMs);
        check(menu, "source_control_missing", "The source-options menu did not open.");
        const sourceAction = menu.querySelector('[data-artifact-action="view-source"]');
        check(sourceAction instanceof HTMLButtonElement &&
          sourceAction.getAttribute("role") === "menuitem", "source_control_missing",
        "The source menu has no semantic View data source action.");
        sourceAction.click();
        const dialog = await waitUntil(() =>
          document.querySelector('[data-artifact-dialog="source"]'), config.actionTimeoutMs);
        check(visible(dialog), "source_dialog_invalid", "The source dialog did not become visible.");
        check(Array.from(dialog.querySelectorAll('[role="tab"]'))
          .some((node) => node.textContent?.trim() === "Overview"),
        "source_dialog_invalid", "The source dialog has no Overview tab.");
        sourceInteraction = "semantic_click";
      }

      send({ counts: actualCounts, ok: true, sourceInteraction });
    } catch (error) {
      send({
        code: error?.code ?? "probe_failed",
        details: error?.details ?? {},
        error: error?.message ?? String(error),
        ok: false,
      });
    }
  };
  void run();
}

function portableHarness(encodedConfig) {
  const decode = (value) => JSON.parse(new TextDecoder().decode(
    Uint8Array.from(atob(value), (character) => character.charCodeAt(0)),
  ));
  const encode = (value) => {
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    let binary = "";
    for (let index = 0; index < bytes.length; index += 8_192) {
      binary += String.fromCharCode(...bytes.subarray(index, index + 8_192));
    }
    return btoa(binary);
  };
  const config = decode(encodedConfig);
  let complete = false;
  const finish = (result) => {
    if (complete) return;
    complete = true;
    const marker = document.createElement("meta");
    marker.id = "codex-portable-verifier-result";
    marker.dataset.result = encode(result);
    document.head.append(marker);
  };
  addEventListener("message", (event) => {
    if (event.data?.channel === config.channel) finish(event.data.result);
  });
  document.querySelector("iframe").srcdoc = config.html;
  setTimeout(() => finish({
    code: "probe_timeout",
    error: `Portable verifier timed out waiting for ${config.viewport.name}.`,
    ok: false,
  }), config.timeoutMs);
}

function buildSingleViewportHarness({ channel, html, timeoutMs, viewport }) {
  const config = encodeJson({ channel, html, timeoutMs, viewport });
  return `<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">` +
    `<iframe title="${viewport.name} report" width="${viewport.width}" height="${viewport.height}" ` +
    `style="border:0;display:block;width:${viewport.width}px;height:${viewport.height}px"></iframe>` +
    `<script>(${portableHarness.toString()})(${JSON.stringify(config)});<\/script>` +
    `</body></html>`;
}

function parseProbeResult(dom) {
  const match = dom.match(
    /<meta id="codex-portable-verifier-result" data-result="([A-Za-z0-9+/=]+)"\s*\/?>/,
  );
  if (!match) throw new Error("Portable verifier produced no result marker.");
  return JSON.parse(Buffer.from(match[1], "base64").toString("utf8"));
}

async function verify(options = {}) {
  const structure = verifyPortableArtifactStructure(options);
  const html = readFileSync(options.htmlPath, "utf8");
  const directory = mkdtempSync(join(tmpdir(), "trecvid-report-verifier-"));
  const viewportResults = [];
  const viewports = [
    { name: "desktop", width: 1_440, height: 1_000 },
    { name: "mobile", width: 390, height: 844 },
  ];

  try {
    for (const viewport of viewports) {
      const channel = randomUUID();
      const probeConfig = {
        actionTimeoutMs: options.actionTimeoutMs,
        channel,
        checkSource: viewport.name === "desktop",
        expectedCounts: structure.counts,
        readerRoot: "#data-analytics-portable-reader",
        readyTimeoutMs: options.readyTimeoutMs,
        title: structure.title,
        viewport,
      };
      const closingBody = html.lastIndexOf("</body>");
      if (closingBody < 0) throw new Error("Portable HTML has no closing body tag.");
      const probeScript = `<script>(${portableReportProbe.toString()})` +
        `(${JSON.stringify(encodeJson(probeConfig))});<\/script>`;
      const instrumentedHtml = `${html.slice(0, closingBody)}${probeScript}` +
        `${html.slice(closingBody)}`;
      const probeTimeoutMs = options.readyTimeoutMs +
        (viewport.name === "desktop" ? options.actionTimeoutMs : 0) + 1_000;
      const harnessPath = join(directory, `${viewport.name}.html`);
      const harnessHtml = buildSingleViewportHarness({
        channel,
        html: instrumentedHtml,
        timeoutMs: probeTimeoutMs,
        viewport,
      });
      writeFileSync(harnessPath, harnessHtml, "utf8");

      const browserResult = await spawnChromiumDump({
        arguments: chromiumDumpArguments({
          height: Math.max(1_000, viewport.height) + 143,
          profilePath: join(directory, `${viewport.name}-profile`),
          url: pathToFileURL(harnessPath).href,
          virtualTimeBudgetMs: probeTimeoutMs + 1_000,
          width: Math.max(800, viewport.width + 16),
        }),
        executablePath: browserExecutable,
        timeoutMs: options.timeoutMs,
      });
      const result = parseProbeResult(browserResult.stdout);
      if (!result?.ok) {
        const detail = result?.details && Object.keys(result.details).length
          ? ` ${JSON.stringify(result.details)}` : "";
        const error = new Error((result?.error ??
          `Portable verification failed at ${viewport.name}.`) + detail);
        error.code = result?.code ?? "verification_failed";
        throw error;
      }
      viewportResults.push(result);
    }

    return {
      ...structure,
      sourceDialog: "passed",
      sourceInteraction: viewportResults[0].sourceInteraction,
      timings: {},
      viewports: viewports.map(({ width }) => width),
    };
  } finally {
    rmSync(directory, { force: true, recursive: true });
  }
}

try {
  const result = await deliverPortableArtifact({
    actionTimeoutMs: 5_000,
    inputPath: new URL("./artifact.json", import.meta.url).pathname,
    outputPath: new URL("./report.html", import.meta.url).pathname,
    readyTimeoutMs: 15_000,
    screenshotPath: new URL("./verification-failure.png", import.meta.url).pathname,
    timeoutMs: 30_000,
  }, { build, extract, verify });
  process.stdout.write(`${JSON.stringify(result)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify(error?.deliveryResult ?? {
    ok: false,
    code: error?.code ?? "delivery_failed",
    error: error?.message ?? String(error),
  })}\n`);
  process.exitCode = 1;
}
