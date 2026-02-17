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

function hrefWithRoot(rootPrefix: string, target: string): string {
  return `${rootPrefix}${target}`.replace(/\/{2,}/g, "/");
}

function renderDocNavigation(input: DocumentTemplateInput): string {
  return input.docs
    .map((doc) => {
      const isActive = doc.outputRelPath === input.current.outputRelPath;
      const classNames = isActive ? "doc-link is-active" : "doc-link";
      const href = hrefWithRoot(input.rootPrefix, doc.outputRelPath);

      return `<a class="${classNames}" data-doc-link href="${escapeHtml(href)}"><span>${escapeHtml(
        doc.title
      )}</span><small>${escapeHtml(readableSourcePath(doc.relativeMarkdownPath))}</small></a>`;
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
  const stylesHref = hrefWithRoot(input.rootPrefix, "assets/styles.css");
  const appHref = hrefWithRoot(input.rootPrefix, "assets/app.js");
  const katexHref = hrefWithRoot(input.rootPrefix, "assets/vendor/katex/katex.min.css");
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
  const list = input.docs
    .map(
      (doc) =>
        `<a class="landing-link" href="${escapeHtml(doc.outputRelPath)}"><strong>${escapeHtml(
          doc.title
        )}</strong><span>${escapeHtml(doc.relativeMarkdownPath)}</span></a>`
    )
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
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body class="landing" data-theme="ivory">
  <div class="ambient-layer ambient-layer-a" aria-hidden="true"></div>
  <div class="ambient-layer ambient-layer-b" aria-hidden="true"></div>
  <main class="landing-main">
    <p class="landing-kicker">Generated Documentation</p>
    <h1>${escapeHtml(input.siteTitle)}</h1>
    <p class="landing-meta">Source: <code>${escapeHtml(input.sourceDirectory)}</code> | Built: <time datetime="${escapeHtml(
    input.generatedAt
  )}">${escapeHtml(input.generatedAt)}</time></p>
    <section class="landing-grid">
${list}
    </section>
  </main>
</body>
</html>`;
}
