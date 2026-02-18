import { promises as fs } from "node:fs";
import path from "node:path";
import { build as esbuild } from "esbuild";
import { createMarkdownRenderer, extractPrimaryTitle, normalizeMathSyntax } from "./markdown.js";
import { renderDocumentTemplate, renderLandingTemplate } from "./template.js";
import type { DocumentPage, RenderEnv } from "./types.js";

const DEFAULT_SOURCE_DIR = "distribution";
const DEFAULT_OUTPUT_DIR = "site";
const DEFAULT_SITE_TITLE = "AI Technical Documentation";
const DEFAULT_LANDING_KICKER = "Technical Knowledge Base";
const DEFAULT_LANDING_SUBTITLE =
  "A scalable documentation workspace for LLMs, agentic AI, and advanced engineering topics.";
const SITE_CONFIG_FILENAME = "site.config.json";
const EXTERNAL_PROTOCOL_RE = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i;

interface SiteConfig {
  siteTitle: string;
  landingKicker: string;
  landingSubtitle: string;
}

interface SiteConfigFile {
  siteTitle?: unknown;
  landingKicker?: unknown;
  landingSubtitle?: unknown;
}

function toPosixPath(value: string): string {
  return value.split(path.sep).join("/");
}

function normalizeRelativeDir(rootDir: string, targetDir: string): string {
  const relative = toPosixPath(path.relative(rootDir, targetDir));
  return relative.length > 0 ? relative : ".";
}

function rewriteMarkdownHref(href: string): string {
  if (!href || href.startsWith("#") || EXTERNAL_PROTOCOL_RE.test(href)) {
    return href;
  }

  const hashIndex = href.indexOf("#");
  const queryIndex = href.indexOf("?");
  let splitIndex = href.length;

  if (hashIndex >= 0) {
    splitIndex = Math.min(splitIndex, hashIndex);
  }

  if (queryIndex >= 0) {
    splitIndex = Math.min(splitIndex, queryIndex);
  }

  const pathname = href.slice(0, splitIndex);
  const suffix = href.slice(splitIndex);

  if (!pathname.toLowerCase().endsWith(".md")) {
    return href;
  }

  return `${pathname.slice(0, -3)}.html${suffix}`;
}

function rewriteRenderedHtmlLinks(html: string): string {
  return html.replace(
    /\b(href|src)=("([^"]+)"|'([^']+)')/gi,
    (segment, attribute: string, _rawQuoted: string, doubleQuoted: string | undefined, singleQuoted: string | undefined) => {
      const quote = doubleQuoted !== undefined ? '"' : "'";
      const rawValue = doubleQuoted ?? singleQuoted ?? "";
      const rewritten = rewriteMarkdownHref(rawValue);

      if (rewritten === rawValue) {
        return segment;
      }

      return `${attribute}=${quote}${rewritten}${quote}`;
    }
  );
}

async function ensureDirectory(targetDir: string): Promise<void> {
  await fs.mkdir(targetDir, { recursive: true });
}

async function collectMarkdownFiles(sourceDir: string): Promise<string[]> {
  const discovered: string[] = [];

  async function walk(currentDir: string): Promise<void> {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        await walk(fullPath);
        continue;
      }

      if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
        discovered.push(fullPath);
      }
    }
  }

  await walk(sourceDir);
  discovered.sort((left, right) => toPosixPath(left).localeCompare(toPosixPath(right)));
  return discovered;
}

async function copyDirectory(sourceDir: string, targetDir: string): Promise<void> {
  await ensureDirectory(targetDir);
  const entries = await fs.readdir(sourceDir, { withFileTypes: true });

  for (const entry of entries) {
    const sourcePath = path.join(sourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);

    if (entry.isDirectory()) {
      await copyDirectory(sourcePath, targetPath);
      continue;
    }

    if (entry.isFile()) {
      await fs.copyFile(sourcePath, targetPath);
    }
  }
}

async function copyNonMarkdownAssets(sourceDir: string, outputDir: string): Promise<void> {
  async function walk(currentDir: string): Promise<void> {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const sourcePath = path.join(currentDir, entry.name);
      const relativePath = toPosixPath(path.relative(sourceDir, sourcePath));
      const targetPath = path.join(outputDir, relativePath);

      if (entry.isDirectory()) {
        await walk(sourcePath);
        continue;
      }

      if (!entry.isFile() || entry.name.toLowerCase().endsWith(".md")) {
        continue;
      }

      await ensureDirectory(path.dirname(targetPath));
      await fs.copyFile(sourcePath, targetPath);
    }
  }

  await walk(sourceDir);
}

function normalizeConfigText(value: unknown, fallback: string): string {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().replace(/\s+/g, " ");
  return normalized.length > 0 ? normalized : fallback;
}

async function loadSiteConfig(sourceDir: string): Promise<SiteConfig> {
  const configPath = path.join(sourceDir, SITE_CONFIG_FILENAME);

  try {
    const raw = await fs.readFile(configPath, "utf8");
    const parsed = JSON.parse(raw) as SiteConfigFile;

    return {
      siteTitle: normalizeConfigText(parsed.siteTitle, DEFAULT_SITE_TITLE),
      landingKicker: normalizeConfigText(parsed.landingKicker, DEFAULT_LANDING_KICKER),
      landingSubtitle: normalizeConfigText(parsed.landingSubtitle, DEFAULT_LANDING_SUBTITLE)
    };
  } catch (error: unknown) {
    if ((error as NodeJS.ErrnoException)?.code === "ENOENT") {
      return {
        siteTitle: DEFAULT_SITE_TITLE,
        landingKicker: DEFAULT_LANDING_KICKER,
        landingSubtitle: DEFAULT_LANDING_SUBTITLE
      };
    }

    throw new Error(`Failed to parse ${SITE_CONFIG_FILENAME}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function relativeAssetRoot(outputRelPath: string): string {
  const depth = outputRelPath.split("/").length - 1;
  return depth === 0 ? "./" : "../".repeat(depth);
}

async function buildClientBundle(rootDir: string, outputDir: string): Promise<void> {
  await esbuild({
    entryPoints: [path.join(rootDir, "src/client.ts")],
    outfile: path.join(outputDir, "assets/app.js"),
    bundle: true,
    minify: true,
    format: "esm",
    target: ["es2022"],
    platform: "browser",
    legalComments: "none",
    logLevel: "silent"
  });
}

async function copyStyles(rootDir: string, outputDir: string): Promise<void> {
  await ensureDirectory(path.join(outputDir, "assets"));
  await fs.copyFile(path.join(rootDir, "src/styles.css"), path.join(outputDir, "assets/styles.css"));
}

async function copyKatexAssets(rootDir: string, outputDir: string): Promise<void> {
  const katexDistDir = path.join(rootDir, "node_modules/katex/dist");
  const katexCssSource = path.join(katexDistDir, "katex.min.css");
  const katexFontsSource = path.join(katexDistDir, "fonts");
  const targetDir = path.join(outputDir, "assets/vendor/katex");
  const targetFonts = path.join(targetDir, "fonts");

  await ensureDirectory(targetDir);
  await fs.copyFile(katexCssSource, path.join(targetDir, "katex.min.css"));
  await copyDirectory(katexFontsSource, targetFonts);
}

async function loadDocuments(sourceDir: string): Promise<DocumentPage[]> {
  const markdownFiles = await collectMarkdownFiles(sourceDir);

  if (markdownFiles.length === 0) {
    throw new Error(`No Markdown files found in ${sourceDir}`);
  }

  const markdown = createMarkdownRenderer();

  const pages = await Promise.all(
    markdownFiles.map(async (sourcePath): Promise<DocumentPage> => {
      const relativeMarkdownPath = toPosixPath(path.relative(sourceDir, sourcePath));
      const outputRelPath = relativeMarkdownPath.replace(/\.md$/i, ".html");
      const rawMarkdown = await fs.readFile(sourcePath, "utf8");
      const normalizedMarkdown = normalizeMathSyntax(rawMarkdown);
      const env: RenderEnv = { headings: [] };
      const html = rewriteRenderedHtmlLinks(markdown.render(normalizedMarkdown, env));
      const title =
        env.headings.find((heading) => heading.level === 1)?.text ??
        extractPrimaryTitle(rawMarkdown, relativeMarkdownPath);

      return {
        sourcePath,
        relativeMarkdownPath,
        outputRelPath,
        title,
        html,
        headings: env.headings
      };
    })
  );

  pages.sort((left, right) => left.outputRelPath.localeCompare(right.outputRelPath));
  return pages;
}

async function writeDocumentPages(
  outputDir: string,
  siteTitle: string,
  sourceDirectoryLabel: string,
  docs: DocumentPage[],
  generatedAt: string
): Promise<void> {
  for (let index = 0; index < docs.length; index += 1) {
    const current = docs[index];
    const previous = index > 0 ? docs[index - 1] : undefined;
    const next = index + 1 < docs.length ? docs[index + 1] : undefined;
    const rootPrefix = relativeAssetRoot(current.outputRelPath);

    const html = renderDocumentTemplate({
      siteTitle,
      sourceDirectory: sourceDirectoryLabel,
      generatedAt,
      rootPrefix,
      docs,
      current,
      previous,
      next
    });

    const targetPath = path.join(outputDir, current.outputRelPath);
    await ensureDirectory(path.dirname(targetPath));
    await fs.writeFile(targetPath, html, "utf8");
  }
}

async function writeLandingPage(
  outputDir: string,
  siteConfig: SiteConfig,
  sourceDirectoryLabel: string,
  docs: DocumentPage[],
  generatedAt: string
): Promise<void> {
  const html = renderLandingTemplate({
    siteTitle: siteConfig.siteTitle,
    landingKicker: siteConfig.landingKicker,
    landingSubtitle: siteConfig.landingSubtitle,
    sourceDirectory: sourceDirectoryLabel,
    generatedAt,
    docs
  });

  await fs.writeFile(path.join(outputDir, "index.html"), html, "utf8");
}

async function main(): Promise<void> {
  const rootDir = process.cwd();
  const sourceArg = process.argv[2] ?? DEFAULT_SOURCE_DIR;
  const outputArg = process.argv[3] ?? DEFAULT_OUTPUT_DIR;
  const sourceDir = path.resolve(rootDir, sourceArg);
  const outputDir = path.resolve(rootDir, outputArg);
  const sourceLabel = normalizeRelativeDir(rootDir, sourceDir);
  const siteConfig = await loadSiteConfig(sourceDir);
  const docs = await loadDocuments(sourceDir);
  const generatedAt = new Date().toISOString();

  await fs.rm(outputDir, { recursive: true, force: true });
  await ensureDirectory(outputDir);

  await Promise.all([
    buildClientBundle(rootDir, outputDir),
    copyStyles(rootDir, outputDir),
    copyKatexAssets(rootDir, outputDir),
    copyNonMarkdownAssets(sourceDir, outputDir)
  ]);

  await writeDocumentPages(outputDir, siteConfig.siteTitle, sourceLabel, docs, generatedAt);
  await writeLandingPage(outputDir, siteConfig, sourceLabel, docs, generatedAt);

  console.log(`Generated ${docs.length} pages in ${outputArg}`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Site generation failed: ${message}`);
  process.exitCode = 1;
});
