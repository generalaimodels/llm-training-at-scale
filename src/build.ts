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
  "A scalable documentation workspace for LLMs, and advanced engineering topics.";
const SITE_CONFIG_FILENAME = "site.config.json";
const EXTERNAL_PROTOCOL_RE = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i;
const NATURAL_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

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

function naturalCompare(left: string, right: string): number {
  return NATURAL_COLLATOR.compare(left, right);
}

function stripMarkdownExtension(relativeMarkdownPath: string): string {
  return relativeMarkdownPath.replace(/\.md$/i, "");
}

function folderFromRelativePath(relativeMarkdownPath: string): string {
  const normalized = stripMarkdownExtension(relativeMarkdownPath);
  const separatorIndex = normalized.lastIndexOf("/");
  return separatorIndex < 0 ? "" : normalized.slice(0, separatorIndex);
}

function baseNameFromRelativePath(relativeMarkdownPath: string): string {
  const normalized = stripMarkdownExtension(relativeMarkdownPath);
  const separatorIndex = normalized.lastIndexOf("/");
  return separatorIndex < 0 ? normalized : normalized.slice(separatorIndex + 1);
}

function parseLeadingOrderSequence(value: string): number[] | null {
  const normalized = value.trim();

  if (!normalized) {
    return null;
  }

  const chapterMatch = normalized.match(/^(?:chapter|section|part|unit|lesson)\s+(\d+(?:\.\d+)*)\b/i);
  const numericMatch = normalized.match(/^(\d+(?:\.\d+)*)\b/);
  const source = chapterMatch?.[1] ?? numericMatch?.[1];

  if (!source) {
    return null;
  }

  const sequence = source
    .split(".")
    .map((segment) => Number.parseInt(segment, 10))
    .filter((segment) => Number.isFinite(segment));

  return sequence.length > 0 ? sequence : null;
}

function compareOrderSequences(left: number[] | null, right: number[] | null): number {
  if (left && right) {
    const limit = Math.min(left.length, right.length);

    for (let index = 0; index < limit; index += 1) {
      const delta = left[index] - right[index];

      if (delta !== 0) {
        return delta;
      }
    }

    return left.length - right.length;
  }

  if (left) {
    return -1;
  }

  if (right) {
    return 1;
  }

  return 0;
}

function isIndexLike(baseName: string, title: string): boolean {
  const normalizedBase = baseName.toLowerCase();
  return (
    normalizedBase === "index" ||
    normalizedBase.endsWith("_index") ||
    normalizedBase.endsWith("-index") ||
    /\bindex\b/i.test(title)
  );
}

function sortDocuments(pages: DocumentPage[]): DocumentPage[] {
  const meta = new Map<
    DocumentPage,
    {
      folder: string;
      rootRank: number;
      isIndex: boolean;
      sequence: number[] | null;
      title: string;
      stemPath: string;
    }
  >();

  for (const page of pages) {
    const stemPath = stripMarkdownExtension(page.relativeMarkdownPath);
    const folder = folderFromRelativePath(page.relativeMarkdownPath);
    const baseName = baseNameFromRelativePath(page.relativeMarkdownPath);

    meta.set(page, {
      folder,
      rootRank: folder.length === 0 ? 0 : 1,
      isIndex: isIndexLike(baseName, page.title),
      sequence: parseLeadingOrderSequence(page.title) ?? parseLeadingOrderSequence(baseName),
      title: page.title,
      stemPath
    });
  }

  const ordered = [...pages];
  ordered.sort((left, right) => {
    const leftMeta = meta.get(left);
    const rightMeta = meta.get(right);

    if (!leftMeta || !rightMeta) {
      return naturalCompare(left.outputRelPath, right.outputRelPath);
    }

    if (leftMeta.rootRank !== rightMeta.rootRank) {
      return leftMeta.rootRank - rightMeta.rootRank;
    }

    const folderCompare = naturalCompare(leftMeta.folder, rightMeta.folder);

    if (folderCompare !== 0) {
      return folderCompare;
    }

    if (leftMeta.isIndex !== rightMeta.isIndex) {
      return leftMeta.isIndex ? -1 : 1;
    }

    const sequenceCompare = compareOrderSequences(leftMeta.sequence, rightMeta.sequence);

    if (sequenceCompare !== 0) {
      return sequenceCompare;
    }

    const titleCompare = naturalCompare(leftMeta.title, rightMeta.title);

    if (titleCompare !== 0) {
      return titleCompare;
    }

    return naturalCompare(leftMeta.stemPath, rightMeta.stemPath);
  });

  return ordered;
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

  return sortDocuments(pages);
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
