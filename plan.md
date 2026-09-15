# Plan: Deep Audit & Fact-Check of https://compatible-ai-zeta.vercel.app/

## Goal
Full-site audit: crawl everything, catalog every claim, brutally fact-check, produce (1) correct vs incorrect list, (2) required go-live changes, (3) improvement recommendations.

## Stage 1 — Crawl & Catalog (Explore agent + my own browser pass)
- Visit the site with browser tools; capture full text of every page/route, screenshots.
- Discover all internal links/routes (nav, footer, anchors) and crawl each.
- Output: complete content inventory file (/mnt/agents/output/site_crawl.md) listing every claim, feature statement, stat, pricing detail, testimonial, comparison, legal/policy text.

## Stage 2 — Fact-Check (Verifier agents, parallel)
- Split the claim inventory into batches; verifier subagents check each claim against authoritative sources (official docs of referenced tools/standards, web search).
- Each claim tagged: TRUE / FALSE / UNVERIFIABLE / MISLEADING, with evidence.
- Output: /mnt/agents/output/factcheck.md

## Stage 3 — Technical & UX Review (Reviewer agent)
- From crawl data + screenshots: broken links, placeholder text, dead CTAs, missing legal pages (privacy, terms), accessibility/SEO basics, console-visible issues, performance red flags, branding consistency.
- Output: /mnt/agents/output/tech_review.md

## Stage 4 — Synthesize & Deliver (Orchestrator)
- Merge into one brutal report: Correct / Incorrect / Unverifiable, Go-live blockers (must-fix), Recommended improvements.
- Deliver as .md + .docx in /mnt/agents/output/.
