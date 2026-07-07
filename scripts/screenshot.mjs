// Dev-only helper: captures a frame of the running dev server with system
// Chrome so the shader can be visually verified without a display attached.
// Usage: node scripts/screenshot.mjs [url] [outfile] [settleMs] [dragX,dragY]
// The optional drag argument simulates a mouse drag (in pixels) before the
// capture, exercising the orbit-camera pointer pipeline.
import puppeteer from "puppeteer-core";

const url = process.argv[2] ?? "http://localhost:5173";
const out = process.argv[3] ?? "screenshot.png";
const settleMs = Number(process.argv[4] ?? 2000);
const drag = process.argv[5]?.split(",").map(Number);

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: ["--use-angle=metal", "--enable-webgl2", "--window-size=1280,720"],
});

const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });
const consoleMessages = [];
page.on("console", (msg) => consoleMessages.push(`[${msg.type()}] ${msg.text()}`));
page.on("pageerror", (err) => consoleMessages.push(`[pageerror] ${err.message}`));

await page.goto(url, { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, settleMs));

if (drag?.length === 2) {
  const [dx, dy] = drag;
  await page.mouse.move(640, 360);
  await page.mouse.down();
  for (let i = 1; i <= 20; i++) {
    await page.mouse.move(640 + (dx * i) / 20, 360 + (dy * i) / 20);
    await new Promise((r) => setTimeout(r, 16));
  }
  await page.mouse.up();
  // Let the damped camera settle before capturing.
  await new Promise((r) => setTimeout(r, 2500));
}
await page.screenshot({ path: out });
await browser.close();

if (consoleMessages.length) {
  console.log("Console output:");
  for (const m of consoleMessages) console.log("  " + m);
} else {
  console.log("No console errors.");
}
console.log(`Saved ${out}`);
