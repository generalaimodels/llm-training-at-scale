import MarkdownIt from "markdown-it";
import anchor from "markdown-it-anchor";
import * as emojiModule from "markdown-it-emoji";
import taskLists from "markdown-it-task-lists";
import texmath from "markdown-it-texmath";
import hljs from "highlight.js";
import katex from "katex";
import type { Heading, RenderEnv } from "./types.js";

const markdownItEmoji =
  (emojiModule as unknown as { full?: (md: MarkdownIt) => void }).full ??
  (emojiModule as unknown as { default?: (md: MarkdownIt) => void }).default ??
  (emojiModule as unknown as (md: MarkdownIt) => void);

const EXTERNAL_PROTOCOL_RE = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i;

const MATH_BLOCK_ENV_MAP = new Map<string, string | null>([
  ["equation", null],
  ["equation*", null],
  ["align", "aligned"],
  ["align*", "aligned"],
  ["gather", "gathered"],
  ["gather*", "gathered"],
  ["multline", "aligned"],
  ["multline*", "aligned"]
]);

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sanitizeForSlug(value: string): string {
  return value
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/&[a-z]+;/gi, "")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
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

function extractTextFromHeading(raw: string): string {
  return raw
    .replace(/\$([^$\n]+)\$/g, "$1")
    .replace(/\\\(([^)\n]+)\\\)/g, "$1")
    .replace(/\\\[([^\]\n]+)\\\]/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .trim();
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeEscapedInlineMathInLine(line: string): string {
  const parts = line.split(/(`+[^`]*`+)/g);

  for (let idx = 0; idx < parts.length; idx += 1) {
    const segment = parts[idx];

    if (segment.startsWith("`") && segment.endsWith("`")) {
      continue;
    }

    const normalized = segment
      .replace(/\\\$\$/g, "$$")
      .replace(/\\\$([^\n$]+?)\\\$/g, "$$$1$");

    parts[idx] = normalized.replace(/\$\s+([^$\n]*?)\s*\$/g, (_match, expression: string) => `$${expression.trim()}$`);
  }

  return parts.join("");
}

function normalizeEscapedMathDelimiters(markdown: string): string {
  const lines = markdown.split(/\r?\n/);
  const output: string[] = [];
  let inFence = false;
  let fenceDelimiter = "";

  for (const line of lines) {
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);

    if (fenceMatch) {
      const delimiter = fenceMatch[1];

      if (!inFence) {
        inFence = true;
        fenceDelimiter = delimiter;
      } else if (delimiter.startsWith(fenceDelimiter[0]) && delimiter.length >= fenceDelimiter.length) {
        inFence = false;
        fenceDelimiter = "";
      }

      output.push(line);
      continue;
    }

    output.push(inFence ? line : normalizeEscapedInlineMathInLine(line));
  }

  return output.join("\n");
}

function normalizeStandaloneMathBlocks(markdown: string): string {
  const lines = markdown.split(/\r?\n/);
  const result: string[] = [];

  let index = 0;
  let inFence = false;
  let fenceDelimiter = "";

  while (index < lines.length) {
    const currentLine = lines[index];
    const fenceMatch = currentLine.match(/^\s*(`{3,}|~{3,})/);

    if (fenceMatch) {
      const delimiter = fenceMatch[1];

      if (!inFence) {
        inFence = true;
        fenceDelimiter = delimiter;
      } else if (delimiter.startsWith(fenceDelimiter[0]) && delimiter.length >= fenceDelimiter.length) {
        inFence = false;
        fenceDelimiter = "";
      }

      result.push(currentLine);
      index += 1;
      continue;
    }

    if (!inFence) {
      const beginMatch = currentLine.match(/^\s*\\begin\{([a-zA-Z*]+)\}\s*$/);

      if (beginMatch) {
        const envName = beginMatch[1];
        const mappedEnv = MATH_BLOCK_ENV_MAP.get(envName);

        if (mappedEnv !== undefined) {
          const blockLines: string[] = [];
          let innerIndex = index + 1;
          let foundEnd = false;
          const endPattern = new RegExp(`^\\s*\\\\end\\{${escapeRegExp(envName)}\\}\\s*$`);

          while (innerIndex < lines.length) {
            const probe = lines[innerIndex];

            if (endPattern.test(probe)) {
              foundEnd = true;
              break;
            }

            blockLines.push(probe);
            innerIndex += 1;
          }

          if (foundEnd) {
            result.push("$$");

            if (mappedEnv === null) {
              result.push(...blockLines);
            } else {
              result.push(`\\begin{${mappedEnv}}`);
              result.push(...blockLines);
              result.push(`\\end{${mappedEnv}}`);
            }

            result.push("$$");
            index = innerIndex + 1;
            continue;
          }
        }
      }
    }

    result.push(currentLine);
    index += 1;
  }

  return result.join("\n");
}

export function extractPrimaryTitle(markdown: string, fallbackPath: string): string {
  const lines = markdown.split(/\r?\n/);

  for (let i = 0; i < lines.length; i += 1) {
    const current = lines[i];
    const atxMatch = current.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/);

    if (atxMatch) {
      return extractTextFromHeading(atxMatch[1]);
    }

    if (i + 1 < lines.length) {
      const next = lines[i + 1];
      const setextMatch = next.match(/^\s*(=+|-+)\s*$/);

      if (setextMatch && current.trim()) {
        return extractTextFromHeading(current);
      }
    }
  }

  const fileName = fallbackPath.split("/").at(-1) ?? fallbackPath;
  return fileName
    .replace(/\.md$/i, "")
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();
}

export function normalizeMathSyntax(markdown: string): string {
  return normalizeStandaloneMathBlocks(normalizeEscapedMathDelimiters(markdown));
}

export function createMarkdownRenderer(): MarkdownIt {
  const markdown = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true,
    highlight: (code: string, language: string): string => {
      const normalizedLanguage = language.trim().toLowerCase();
      const languageClass = normalizedLanguage ? ` language-${escapeHtml(normalizedLanguage)}` : "";

      if (normalizedLanguage && hljs.getLanguage(normalizedLanguage)) {
        const highlighted = hljs.highlight(code, {
          language: normalizedLanguage,
          ignoreIllegals: true
        }).value;
        return `<pre class="hljs"><code class="hljs${languageClass}">${highlighted}</code></pre>`;
      }

      return `<pre class="hljs"><code class="hljs${languageClass}">${escapeHtml(code)}</code></pre>`;
    }
  });

  markdown.use(taskLists, { enabled: true, label: true, labelAfter: true });
  markdown.use(markdownItEmoji);
  markdown.use(anchor, {
    slugify: sanitizeForSlug,
    level: [1, 2, 3, 4, 5, 6],
    permalink: false
  });

  markdown.use(texmath, {
    engine: katex,
    delimiters: "dollars",
    katexOptions: { throwOnError: false, strict: "ignore" },
    outerSpace: false
  });

  markdown.use(texmath, {
    engine: katex,
    delimiters: "brackets",
    katexOptions: { throwOnError: false, strict: "ignore" },
    outerSpace: false
  });

  markdown.core.ruler.after("inline", "rewrite-markdown-links", (state): void => {
    for (const token of state.tokens) {
      if (token.type !== "inline" || !token.children) {
        continue;
      }

      for (const child of token.children) {
        if (child.type !== "link_open") {
          continue;
        }

        const href = child.attrGet("href");

        if (!href) {
          continue;
        }

        child.attrSet("href", rewriteMarkdownHref(href));
      }
    }
  });

  markdown.core.ruler.push("collect-headings", (state): void => {
    const env = (state.env ?? {}) as RenderEnv;
    env.headings = [];

    for (let idx = 0; idx < state.tokens.length; idx += 1) {
      const open = state.tokens[idx];

      if (open.type !== "heading_open") {
        continue;
      }

      const inline = state.tokens[idx + 1];

      if (!inline || inline.type !== "inline") {
        continue;
      }

      const text = extractTextFromHeading(inline.content);

      if (!text) {
        continue;
      }

      const level = Number.parseInt(open.tag.slice(1), 10);
      const id = open.attrGet("id") ?? sanitizeForSlug(text);

      env.headings.push({ level, id, text } as Heading);
    }
  });

  const originalFenceRenderer =
    markdown.renderer.rules.fence ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));

  markdown.renderer.rules.fence = (tokens, index, options, env, self): string => {
    const token = tokens[index];
    const language = token.info.trim().split(/\s+/)[0]?.toLowerCase() ?? "";

    if (language === "mermaid") {
      return `<pre class="mermaid">${escapeHtml(token.content)}</pre>`;
    }

    const rendered = originalFenceRenderer(tokens, index, options, env, self);
    return `<div class="code-shell">${rendered}</div>`;
  };

  const originalLinkRenderer =
    markdown.renderer.rules.link_open ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));

  markdown.renderer.rules.link_open = (tokens, index, options, env, self): string => {
    const token = tokens[index];
    const href = token.attrGet("href") ?? "";

    if (/^https?:\/\//i.test(href)) {
      token.attrSet("target", "_blank");
      token.attrSet("rel", "noopener noreferrer");
    }

    return originalLinkRenderer(tokens, index, options, env, self);
  };

  const originalImageRenderer =
    markdown.renderer.rules.image ??
    ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));

  markdown.renderer.rules.image = (tokens, index, options, env, self): string => {
    const token = tokens[index];

    if (!token.attrGet("loading")) {
      token.attrSet("loading", "lazy");
    }

    return originalImageRenderer(tokens, index, options, env, self);
  };

  return markdown;
}
