import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site";
import { getRunPairs, runPath } from "@/lib/runPages";

export default function sitemap(): MetadataRoute.Sitemap {
  const staticRoutes = ["", "/recommend", "/terminology", "/privacy", "/terms"].map(
    (route) => ({
      url: `${SITE_URL}${route}`,
      lastModified: new Date("2026-09-15"),
    })
  );

  const runRoutes = getRunPairs().map(({ model, gpu }) => ({
    url: `${SITE_URL}${runPath(model.id, gpu.id)}`,
    lastModified: new Date("2026-09-15"),
  }));

  return [...staticRoutes, ...runRoutes];
}
