import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site";

export default function sitemap(): MetadataRoute.Sitemap {
  return ["", "/recommend", "/terminology", "/privacy", "/terms"].map((route) => ({
    url: `${SITE_URL}${route}`,
    lastModified: new Date("2026-09-15"),
  }));
}
