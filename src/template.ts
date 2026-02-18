import type { DocumentPage, Heading } from "./types.js";

interface DocumentTemplateInput {
  siteTitle: string;
  sourceDirectory: string;
  generatedAt: string;
  rootPrefix: string;
  docs: DocumentPage[];
  current: DocumentPage;
  previous?: DocumentPage;
  next?: DocumentPage;
}

interface LandingTemplateInput {
  siteTitle: string;
  sourceDirectory: string;
  generatedAt: string;
  docs: DocumentPage[];
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function readableSourcePath(value: string): string {
  return value.replace(/\.md$/i, "");
}

function folderFromRelativePath(relativePath: string): string {
  const normalized = readableSourcePath(relativePath);
  const lastSlash = normalized.lastIndexOf("/");
  return lastSlash <= 0 ? "root" : normalized.slice(0, lastSlash);
}

function hrefWithRoot(rootPrefix: string, target: string): string {
  return `${rootPrefix}${target}`.replace(/\/{2,}/g, "/");
}

function withVersion(href: string, version: string): string {
  return `${href}?v=${encodeURIComponent(version)}`;
}

function renderDocNavigation(input: DocumentTemplateInput): string {
  return input.docs
    .map((doc) => {
      const isActive = doc.outputRelPath === input.current.outputRelPath;
      const classNames = isActive ? "doc-link is-active" : "doc-link";
      const href = hrefWithRoot(input.rootPrefix, doc.outputRelPath);
      const readablePath = readableSourcePath(doc.relativeMarkdownPath);
      const folderLabel = folderFromRelativePath(doc.relativeMarkdownPath);
      const titleSearch = doc.title.toLowerCase();
      const pathSearch = readablePath.toLowerCase();
      const folderSearch = folderLabel.toLowerCase();

      return `<a class="${classNames}" data-doc-link data-doc-title="${escapeHtml(titleSearch)}" data-doc-path="${escapeHtml(
        pathSearch
      )}" data-doc-folder="${escapeHtml(folderSearch)}" href="${escapeHtml(href)}"><span>${escapeHtml(
        doc.title
      )}</span><small>${escapeHtml(readablePath)}</small></a>`;
    })
    .join("\n");
}

function renderToc(headings: Heading[]): string {
  const relevant = headings.filter((heading) => heading.level >= 2 && heading.level <= 4);

  if (relevant.length === 0) {
    return `<p class="toc-empty">No section anchors found in this page.</p>`;
  }

  return relevant
    .map((heading) => {
      const cls = `toc-link toc-level-${heading.level}`;
      return `<a class="${cls}" data-toc-link href="#${escapeHtml(heading.id)}">${escapeHtml(
        heading.text
      )}</a>`;
    })
    .join("\n");
}

function renderPagerLink(rootPrefix: string, label: string, doc?: DocumentPage): string {
  if (!doc) {
    return `<span class="pager-link is-empty">${escapeHtml(label)}</span>`;
  }

  const href = hrefWithRoot(rootPrefix, doc.outputRelPath);

  return `<a class="pager-link" href="${escapeHtml(href)}"><small>${escapeHtml(label)}</small><strong>${escapeHtml(
    doc.title
  )}</strong></a>`;
}

export function renderDocumentTemplate(input: DocumentTemplateInput): string {
  const stylesHref = withVersion(hrefWithRoot(input.rootPrefix, "assets/styles.css"), input.generatedAt);
  const appHref = withVersion(hrefWithRoot(input.rootPrefix, "assets/app.js"), input.generatedAt);
  const katexHref = withVersion(hrefWithRoot(input.rootPrefix, "assets/vendor/katex/katex.min.css"), input.generatedAt);
  const pageTitle = `${input.current.title} | ${input.siteTitle}`;

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeHtml(input.current.title)}">
  <title>${escapeHtml(pageTitle)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="${escapeHtml(katexHref)}">
  <link rel="stylesheet" href="${escapeHtml(stylesHref)}">
</head>
<body data-theme="ivory">
  <div class="ambient-layer ambient-layer-a" aria-hidden="true"></div>
  <div class="ambient-layer ambient-layer-b" aria-hidden="true"></div>
  <header class="topbar">
    <button id="menu-toggle" class="icon-btn" type="button" aria-label="Toggle navigation">Menu</button>
    <a class="brand" href="${escapeHtml(hrefWithRoot(input.rootPrefix, "index.html"))}">
      <span class="brand-kicker">Docs</span>
      <strong>${escapeHtml(input.siteTitle)}</strong>
    </a>
    <div class="topbar-actions">
      <input id="nav-filter" class="doc-filter" type="search" placeholder="Filter documents" aria-label="Filter documents">
      <p id="filter-status" class="filter-status" aria-live="polite"></p>
      <button id="theme-toggle" class="icon-btn" type="button" aria-label="Switch theme">Graphite</button>
    </div>
  </header>
  <div class="site-shell">
    <aside class="nav-pane" id="left-nav">
      <p class="pane-label">Source Folder</p>
      <p class="pane-title">${escapeHtml(input.sourceDirectory)}</p>
      <nav class="doc-nav" aria-label="Document list">
${renderDocNavigation(input)}
      </nav>
    </aside>
    <main class="content-pane">
      <article class="doc-content">
${input.current.html}
      </article>
      <section class="doc-pager" aria-label="Document pagination">
        ${renderPagerLink(input.rootPrefix, "Previous", input.previous)}
        ${renderPagerLink(input.rootPrefix, "Next", input.next)}
      </section>
      <footer class="doc-footer">
        <p>Generated from <code>${escapeHtml(input.sourceDirectory)}</code> at <time datetime="${escapeHtml(
          input.generatedAt
        )}">${escapeHtml(input.generatedAt)}</time>.</p>
      </footer>
    </main>
    <aside class="toc-pane">
      <p class="pane-label">On This Page</p>
      <nav class="toc-nav" aria-label="Table of contents">
${renderToc(input.current.headings)}
      </nav>
    </aside>
  </div>
  <script type="module" src="${escapeHtml(appHref)}"></script>
</body>
</html>`;
}

export function renderLandingTemplate(input: LandingTemplateInput): string {
  const stylesHref = withVersion("assets/styles.css", input.generatedAt);
  const appHref = withVersion("assets/app.js", input.generatedAt);
  const docCount = input.docs.length;
  const folderCounts = new Map<string, number>();

  for (const doc of input.docs) {
    const folder = folderFromRelativePath(doc.relativeMarkdownPath);
    folderCounts.set(folder, (folderCounts.get(folder) ?? 0) + 1);
  }

  const topFolders = Array.from(folderCounts.entries())
    .sort((left, right) => {
      if (right[1] !== left[1]) {
        return right[1] - left[1];
      }

      return left[0].localeCompare(right[0]);
    })
    .slice(0, 8);

  const folderPills = topFolders
    .map(
      ([folder, count]) =>
        `<span class="folder-pill"><b>${escapeHtml(folder)}</b><small>${count} docs</small></span>`
    )
    .join("\n");

  const cards = input.docs
    .map((doc) => {
      const readablePath = readableSourcePath(doc.relativeMarkdownPath);
      const folder = folderFromRelativePath(doc.relativeMarkdownPath);
      const depth = readablePath.split("/").length;

      return `<article class="doc-card" data-landing-card data-doc-title="${escapeHtml(
        doc.title.toLowerCase()
      )}" data-doc-path="${escapeHtml(readablePath.toLowerCase())}" data-doc-folder="${escapeHtml(
        folder.toLowerCase()
      )}">
  <a class="doc-card-link" href="${escapeHtml(doc.outputRelPath)}">
    <p class="doc-card-folder">${escapeHtml(folder)}</p>
    <h2>${escapeHtml(doc.title)}</h2>
    <p class="doc-card-path">${escapeHtml(readablePath)}</p>
    <p class="doc-card-depth">Depth ${depth}</p>
  </a>
</article>`;
    })
    .join("\n");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeHtml(input.siteTitle)}">
  <title>${escapeHtml(input.siteTitle)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="${escapeHtml(stylesHref)}">
</head>
<body class="landing" data-theme="ivory">
  <div class="ambient-layer ambient-layer-a" aria-hidden="true"></div>
  <div class="ambient-layer ambient-layer-b" aria-hidden="true"></div>
  <main class="landing-main">
    <section class="landing-hero">
      <p class="landing-kicker">Distributed Systems Documentation</p>
      <h1>${escapeHtml(input.siteTitle)}</h1>
      <p class="landing-subtitle">Premium-grade reading experience for parallelism architecture, optimization mechanics, and scale-ready model training systems.</p>
      <div class="landing-meta-grid">
        <article class="meta-card">
          <span>Total Documents</span>
          <strong>${docCount}</strong>
        </article>
        <article class="meta-card">
          <span>Source Root</span>
          <strong>${escapeHtml(input.sourceDirectory)}</strong>
        </article>
        <article class="meta-card">
          <span>Generated</span>
          <strong><time datetime="${escapeHtml(input.generatedAt)}">${escapeHtml(input.generatedAt)}</time></strong>
        </article>
      </div>
    </section>

    <section class="landing-intelligence">
      <div class="landing-search-shell">
        <label for="landing-filter" class="landing-search-label">Search docs by title, path, or folder</label>
        <input id="landing-filter" class="landing-search" type="search" placeholder="Try: tensor, pipeline, expert, context, data...">
      </div>
      <p id="landing-results" class="landing-results" aria-live="polite">Showing ${docCount} of ${docCount} documents</p>
      <div class="folder-pills">
${folderPills}
      </div>
    </section>

    <section class="landing-grid" id="landing-grid">
${cards}
    </section>
  </main>
  <script type="module" src="${escapeHtml(appHref)}"></script>
</body>
</html>`;
}
