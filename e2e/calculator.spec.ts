import { test, expect, type Page } from "@playwright/test";

// Selected GPU cards get `ring-1` from ui/Card.tsx — the only ring-1 user.
const selectedCards = (page: Page) => page.locator("div.ring-1");

test.describe("calculator regression suite", () => {
  test("a. share link restores scenario + compare GPU selection", async ({ page }) => {
    await page.goto("/?scenario=compare&model=llama3-8b&cgpu=rtx3090,rtx3060");

    // Compare tab is the active scenario tab
    await expect(page.getByRole("button", { name: "compare", exact: true })).toHaveClass(
      /bg-primary\/20/
    );

    // Both GPUs from the cgpu param show as selected
    await expect(selectedCards(page)).toHaveCount(2);
    await expect(selectedCards(page).filter({ hasText: "RTX 3090" })).toHaveCount(1);
    await expect(selectedCards(page).filter({ hasText: "RTX 3060" })).toHaveCount(1);

    // And they appear as columns in the comparison table
    await expect(page.locator("th", { hasText: "RTX 3090" })).toBeVisible();
    await expect(page.locator("th", { hasText: "RTX 3060" })).toBeVisible();
  });

  test("b. URL state is not clobbered by defaults after hydration", async ({ page }) => {
    await page.goto("/?model=llama3-70b&quant=FP16");

    // Wait until the state->URL writer has run (it adds ctx=, batch=, ...).
    await expect(page).toHaveURL(/[?&]ctx=/);

    // The incoming params must survive — not be rewritten to defaults.
    await expect(page).toHaveURL(/model=llama3-70b/);
    await expect(page).toHaveURL(/quant=FP16/);

    // Model picker shows Llama 3 70B
    await expect(page.locator("select")).toHaveValue("llama3-70b");
    await expect(page.getByText("is estimated to need")).toBeVisible();
  });

  test("c. share button copies and shows feedback without crashing", async ({ page, context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.goto("/");

    const share = page.getByRole("button", { name: "Share", exact: true });
    await expect(share).toBeVisible();
    await share.click();

    await expect(page.getByRole("button", { name: "Copied!" })).toBeVisible();
    // Page is still alive — the calculator readout still renders.
    await expect(page.getByText("is estimated to need")).toBeVisible();
  });

  test("d. compare GPU selection does not leak into inference", async ({ page }) => {
    await page.goto("/");

    // Compare: default 2 GPUs, add a third (RTX 3060)
    await page.getByRole("button", { name: "compare", exact: true }).click();
    await expect(selectedCards(page)).toHaveCount(2);
    await page.locator("div.cursor-pointer", { hasText: "RTX 3060" }).first().click();
    await expect(selectedCards(page)).toHaveCount(3);

    // Inference: still exactly 1 selected GPU
    await page.getByRole("button", { name: "inference", exact: true }).click();
    await expect(selectedCards(page)).toHaveCount(1);
    await expect(selectedCards(page).first()).toContainText("RTX 4090");

    // Back to Compare: still 3
    await page.getByRole("button", { name: "compare", exact: true }).click();
    await expect(selectedCards(page)).toHaveCount(3);
  });

  test("e. compare Estimated TPS is per-GPU, not a placeholder", async ({ page }) => {
    await page.goto("/?scenario=compare&cgpu=rtx4090,rtx3090,rtx3060");

    const tpsRow = page.locator("tr", { hasText: "Estimated TPS" });
    await expect(tpsRow).toBeVisible();
    const cells = (await tpsRow.locator("td").allTextContents()).map((s) => s.trim());
    const values = cells.slice(1); // first cell is the row label
    expect(values.length).toBeGreaterThanOrEqual(3);
    for (const v of values) expect(v).toMatch(/tok\/s/);
    expect(new Set(values).size).toBeGreaterThan(1);
  });

  test("f. CPU offload verdict mentions VRAM and system RAM", async ({ page }) => {
    await page.goto("/");

    await page.getByRole("button", { name: /advanced settings/i }).click();
    await page.getByRole("switch").click();

    const verdict = page.locator("p", { hasText: "is estimated to need" });
    await expect(verdict).toContainText(/VRAM/);
    await expect(verdict).toContainText(/system RAM/i);
  });

  test("g. Hugging Face import (mocked) registers and selects a custom model", async ({ page }) => {
    const config = {
      num_hidden_layers: 32,
      hidden_size: 4096,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      vocab_size: 128256,
      max_position_embeddings: 8192,
      torch_dtype: "bfloat16",
    };
    await page.route("**/huggingface.co/**", async (route) => {
      if (route.request().url().endsWith("config.json")) {
        await route.fulfill({ contentType: "application/json", body: JSON.stringify(config) });
      } else {
        // No safetensors index -> dense param estimate path
        await route.fulfill({ status: 404, body: "not found" });
      }
    });

    await page.goto("/");
    await page.getByPlaceholder("Paste a Hugging Face repo or URL...").fill("test-org/test-model");
    await page.getByRole("button", { name: "Import" }).click();

    // Custom model becomes the selected model
    await expect(page.locator("select")).toHaveValue("hf-custom-test-org-test-model");
    await expect(page.getByText(/unverified — imported from Hugging Face/)).toBeVisible();
    await expect(page.getByText(/params estimated from config/)).toBeVisible();

    // A real (non-zero) weights figure renders in the breakdown
    await expect(page.getByText(/≈ [1-9][\d.]* GB.*TOTAL REQUIRED/)).toBeVisible();
  });

  test("h. /recommend honors budget + useCase from the URL", async ({ page }) => {
    await page.goto("/recommend?budget=1500&useCase=edge");

    // Edge use case is active (not the coding default)
    await expect(page.getByRole("button", { name: "Edge / Lightweight" })).toHaveClass(/bg-primary/);

    // A local recommendation is produced...
    await expect(page.getByText("Optimal Loadout Found")).toBeVisible();

    // ...and the hardware is a single GPU, not a "2x/4x" rig
    const hardwareName = page.locator("div.text-2xl.font-bold").first();
    await expect(hardwareName).toBeVisible();
    const name = ((await hardwareName.textContent()) ?? "").trim();
    expect(name).not.toMatch(/^[24]x\s/i);
    expect(name).toMatch(/^1x\s/);
  });
});
