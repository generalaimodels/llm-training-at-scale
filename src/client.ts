import mermaid from "mermaid";
import renderMathInElement from "katex/contrib/auto-render";

type ThemeMode = "ivory" | "graphite";

const THEME_STORAGE_KEY = "docs-theme-mode";
const FILTER_HIDDEN_CLASS = "is-filter-hidden";

interface SearchEntry {
  element: HTMLElement;
  index: number;
  title: string;
  path: string;
  folder: string;
  titleTokens: string[];
  pathTokens: string[];
  folderTokens: string[];
}

interface RankedEntry {
  entry: SearchEntry;
  score: number;
}

function normalizeSearchQuery(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function tokenizeSearch(value: string): string[] {
  return value.split(/[^\p{L}\p{N}]+/u).filter((token) => token.length > 0);
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  if (target.isContentEditable) {
    return true;
  }

  return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement;
}

function readSearchValue(element: HTMLElement, attributeName: string, fallbackSelector: string): string {
  const attributeValue = element.getAttribute(attributeName);

  if (attributeValue && attributeValue.length > 0) {
    return attributeValue.toLowerCase();
  }

  const fallback = element.querySelector<HTMLElement>(fallbackSelector)?.textContent ?? element.textContent ?? "";
  return fallback.trim().toLowerCase();
}

function buildSearchEntries(elements: HTMLElement[]): SearchEntry[] {
  return elements.map((element, index) => {
    const title = readSearchValue(element, "data-doc-title", "h2, span");
    const path = readSearchValue(element, "data-doc-path", ".doc-card-path, small");
    const folder = readSearchValue(element, "data-doc-folder", ".doc-card-folder");

    return {
      element,
      index,
      title,
      path,
      folder,
      titleTokens: tokenizeSearch(title),
      pathTokens: tokenizeSearch(path),
      folderTokens: tokenizeSearch(folder)
    };
  });
}

function tokenSetScore(queryTokens: string[], candidates: string[]): number {
  let score = 0;

  for (const queryToken of queryTokens) {
    let best = 0;

    for (const candidate of candidates) {
      if (candidate === queryToken) {
        best = Math.max(best, 180);
      } else if (candidate.startsWith(queryToken)) {
        best = Math.max(best, 120);
      } else if (candidate.includes(queryToken)) {
        best = Math.max(best, 64);
      }
    }

    if (best === 0) {
      return -1;
    }

    score += best;
  }

  return score;
}

function computeSearchScore(entry: SearchEntry, normalizedQuery: string, queryTokens: string[]): number {
  if (normalizedQuery.length === 0) {
    return 1;
  }

  let score = -1;

  if (entry.title === normalizedQuery) {
    score = Math.max(score, 1600);
  }

  if (entry.path === normalizedQuery) {
    score = Math.max(score, 1540);
  }

  if (entry.folder === normalizedQuery) {
    score = Math.max(score, 1480);
  }

  if (entry.title.startsWith(normalizedQuery)) {
    score = Math.max(score, 1350);
  }

  if (entry.path.startsWith(normalizedQuery)) {
    score = Math.max(score, 1220);
  }

  if (entry.folder.startsWith(normalizedQuery)) {
    score = Math.max(score, 1120);
  }

  const titleIndex = entry.title.indexOf(normalizedQuery);
  if (titleIndex >= 0) {
    score = Math.max(score, 980 - Math.min(titleIndex, 120));
  }

  const pathIndex = entry.path.indexOf(normalizedQuery);
  if (pathIndex >= 0) {
    score = Math.max(score, 920 - Math.min(pathIndex, 160));
  }

  const folderIndex = entry.folder.indexOf(normalizedQuery);
  if (folderIndex >= 0) {
    score = Math.max(score, 860 - Math.min(folderIndex, 160));
  }

  if (queryTokens.length > 0) {
    const titleTokenScore = tokenSetScore(queryTokens, entry.titleTokens);
    if (titleTokenScore >= 0) {
      score = Math.max(score, 700 + titleTokenScore);
    }

    const pathTokenScore = tokenSetScore(queryTokens, entry.pathTokens);
    if (pathTokenScore >= 0) {
      score = Math.max(score, 650 + pathTokenScore);
    }

    const folderTokenScore = tokenSetScore(queryTokens, entry.folderTokens);
    if (folderTokenScore >= 0) {
      score = Math.max(score, 620 + folderTokenScore);
    }
  }

  return score;
}

function updateFilterStatus(
  statusElement: HTMLElement | null,
  visibleCount: number,
  totalCount: number,
  normalizedQuery: string
): void {
  if (!statusElement) {
    return;
  }

  if (normalizedQuery.length === 0) {
    statusElement.textContent = `${totalCount} docs`;
    return;
  }

  statusElement.textContent = `${visibleCount}/${totalCount} matches`;
}

function applyRankedFilter(
  entries: SearchEntry[],
  container: HTMLElement,
  statusElement: HTMLElement | null,
  rawQuery: string
): void {
  const normalizedQuery = normalizeSearchQuery(rawQuery);
  const queryTokens = tokenizeSearch(normalizedQuery);

  if (normalizedQuery.length === 0) {
    const fragment = document.createDocumentFragment();
    const ordered = [...entries].sort((left, right) => left.index - right.index);

    for (const entry of ordered) {
      entry.element.classList.remove(FILTER_HIDDEN_CLASS);
      fragment.append(entry.element);
    }

    container.append(fragment);
    updateFilterStatus(statusElement, entries.length, entries.length, normalizedQuery);
    return;
  }

  const ranked: RankedEntry[] = [];

  for (const entry of entries) {
    const score = computeSearchScore(entry, normalizedQuery, queryTokens);

    if (score >= 0) {
      ranked.push({ entry, score });
    }
  }

  ranked.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }

    if (left.entry.path.length !== right.entry.path.length) {
      return left.entry.path.length - right.entry.path.length;
    }

    return left.entry.index - right.entry.index;
  });

  const visibleElements = new Set<HTMLElement>();
  const fragment = document.createDocumentFragment();

  for (const item of ranked) {
    visibleElements.add(item.entry.element);
    item.entry.element.classList.remove(FILTER_HIDDEN_CLASS);
    fragment.append(item.entry.element);
  }

  for (const entry of entries) {
    if (!visibleElements.has(entry.element)) {
      entry.element.classList.add(FILTER_HIDDEN_CLASS);
    }
  }

  container.append(fragment);
  updateFilterStatus(statusElement, ranked.length, entries.length, normalizedQuery);
}

function attachSearchShortcut(input: HTMLInputElement): void {
  document.addEventListener("keydown", (event) => {
    if (event.key !== "/" || isTypingTarget(event.target)) {
      return;
    }

    event.preventDefault();
    input.focus();
    input.select();
  });
}

function getThemeButton(): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>("#theme-toggle");
}

function applyTheme(mode: ThemeMode): void {
  document.body.dataset.theme = mode;
  const button = getThemeButton();

  if (button) {
    button.textContent = mode === "ivory" ? "Graphite" : "Ivory";
  }
}

function initThemeToggle(): void {
  const saved = window.localStorage.getItem(THEME_STORAGE_KEY);

  if (saved === "graphite" || saved === "ivory") {
    applyTheme(saved);
  } else {
    applyTheme("ivory");
  }

  const button = getThemeButton();

  if (!button) {
    return;
  }

  button.addEventListener("click", () => {
    const nextMode: ThemeMode = document.body.dataset.theme === "graphite" ? "ivory" : "graphite";
    applyTheme(nextMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, nextMode);
  });
}

function initMenuToggle(): void {
  const menuButton = document.querySelector<HTMLButtonElement>("#menu-toggle");

  if (!menuButton) {
    return;
  }

  menuButton.addEventListener("click", () => {
    document.body.classList.toggle("sidebar-open");
  });
}

function initSidebarDismiss(): void {
  const navPane = document.querySelector<HTMLElement>(".nav-pane");
  const menuButton = document.querySelector<HTMLButtonElement>("#menu-toggle");

  if (!navPane || !menuButton) {
    return;
  }

  const closeSidebar = (): void => {
    document.body.classList.remove("sidebar-open");
  };

  document.addEventListener("click", (event) => {
    if (!document.body.classList.contains("sidebar-open")) {
      return;
    }

    const target = event.target;
    if (!(target instanceof Node)) {
      return;
    }

    if (navPane.contains(target) || menuButton.contains(target)) {
      return;
    }

    closeSidebar();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSidebar();
    }
  });

  window.addEventListener("resize", () => {
    if (window.matchMedia("(min-width: 981px)").matches) {
      closeSidebar();
    }
  });
}

function initDocumentFilter(): void {
  const filter = document.querySelector<HTMLInputElement>("#nav-filter");
  const container = document.querySelector<HTMLElement>(".doc-nav");
  const status = document.querySelector<HTMLElement>("#filter-status");
  const links = Array.from(document.querySelectorAll<HTMLElement>("[data-doc-link]"));

  if (!filter || !container || links.length === 0) {
    return;
  }

  const entries = buildSearchEntries(links);
  let pendingFrame = 0;

  const update = (): void => {
    pendingFrame = 0;
    applyRankedFilter(entries, container, status, filter.value);
  };

  filter.addEventListener("input", () => {
    if (pendingFrame !== 0) {
      window.cancelAnimationFrame(pendingFrame);
    }

    pendingFrame = window.requestAnimationFrame(update);
  });

  attachSearchShortcut(filter);
  update();
}

function initLandingFilter(): void {
  const filter = document.querySelector<HTMLInputElement>("#landing-filter");
  const container = document.querySelector<HTMLElement>("#landing-grid");
  const status = document.querySelector<HTMLElement>("#landing-results");
  const cards = Array.from(document.querySelectorAll<HTMLElement>("[data-landing-card]"));

  if (!filter || !container || cards.length === 0) {
    return;
  }

  const entries = buildSearchEntries(cards);
  let pendingFrame = 0;

  const update = (): void => {
    pendingFrame = 0;
    applyRankedFilter(entries, container, status, filter.value);
  };

  filter.addEventListener("input", () => {
    if (pendingFrame !== 0) {
      window.cancelAnimationFrame(pendingFrame);
    }

    pendingFrame = window.requestAnimationFrame(update);
  });

  attachSearchShortcut(filter);
  update();
}

function initCodeCopyButtons(): void {
  if (!navigator.clipboard?.writeText) {
    return;
  }

  const wrappers = Array.from(document.querySelectorAll<HTMLElement>(".code-shell"));

  for (const wrapper of wrappers) {
    const code = wrapper.querySelector<HTMLElement>("pre code");

    if (!code) {
      continue;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.textContent = "Copy";

    button.addEventListener("click", async () => {
      const raw = code.textContent ?? "";

      if (!raw) {
        return;
      }

      try {
        await navigator.clipboard.writeText(raw);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      } catch {
        button.textContent = "Error";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      }
    });

    wrapper.append(button);
  }
}

function initTableOfContentsHighlight(): void {
  const headings = Array.from(
    document.querySelectorAll<HTMLElement>(".doc-content h1[id], .doc-content h2[id], .doc-content h3[id], .doc-content h4[id]")
  );
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>("[data-toc-link]"));

  if (headings.length === 0 || links.length === 0) {
    return;
  }

  const linkByHash = new Map<string, HTMLAnchorElement>();

  for (const link of links) {
    const href = link.getAttribute("href");

    if (href?.startsWith("#")) {
      linkByHash.set(href.slice(1), link);
    }
  }

  const setActive = (id: string): void => {
    for (const link of links) {
      link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
    }
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => left.boundingClientRect.top - right.boundingClientRect.top);

      if (visible.length > 0) {
        const heading = visible[0].target as HTMLElement;
        setActive(heading.id);
      }
    },
    {
      rootMargin: "-30% 0px -55% 0px",
      threshold: [0, 0.1, 0.8]
    }
  );

  for (const heading of headings) {
    observer.observe(heading);
    const match = linkByHash.get(heading.id);

    if (match) {
      match.addEventListener("click", () => {
        document.body.classList.remove("sidebar-open");
      });
    }
  }
}

function initRevealAnimation(): void {
  const candidates = Array.from(document.querySelectorAll<HTMLElement>(".doc-content > *"));

  if (candidates.length === 0) {
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }

        (entry.target as HTMLElement).classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    {
      rootMargin: "0px 0px -8% 0px",
      threshold: 0.08
    }
  );

  for (const candidate of candidates) {
    candidate.classList.add("reveal");
    observer.observe(candidate);
  }
}

function initMathFallbackRendering(): void {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
      { left: "\\begin{equation}", right: "\\end{equation}", display: true },
      { left: "\\begin{equation*}", right: "\\end{equation*}", display: true },
      { left: "\\begin{align}", right: "\\end{align}", display: true },
      { left: "\\begin{align*}", right: "\\end{align*}", display: true },
      { left: "\\begin{gather}", right: "\\end{gather}", display: true },
      { left: "\\begin{gather*}", right: "\\end{gather*}", display: true },
      { left: "\\begin{multline}", right: "\\end{multline}", display: true },
      { left: "\\begin{multline*}", right: "\\end{multline*}", display: true }
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoredClasses: ["katex", "katex-display"],
    throwOnError: false,
    strict: "ignore"
  });
}

function getMermaidTheme(): "base" | "dark" {
  return document.body.dataset.theme === "graphite" ? "dark" : "base";
}

async function renderMermaidBlock(block: HTMLElement, sequence: number): Promise<void> {
  const source = block.textContent?.trim();

  if (!source) {
    return;
  }

  const mount = document.createElement("figure");
  mount.className = "mermaid-figure";

  const graph = document.createElement("div");
  graph.className = "mermaid-render";
  mount.append(graph);

  try {
    const id = `mermaid-diagram-${sequence}`;
    const rendered = await mermaid.render(id, source);
    graph.innerHTML = rendered.svg;
    block.replaceWith(mount);
  } catch {
    block.classList.add("mermaid-error");
  }
}

async function initMermaid(): Promise<void> {
  const blocks = Array.from(document.querySelectorAll<HTMLElement>("pre.mermaid"));

  if (blocks.length === 0) {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: getMermaidTheme(),
    fontFamily: "Manrope, sans-serif"
  });

  for (let index = 0; index < blocks.length; index += 1) {
    await renderMermaidBlock(blocks[index], index);
  }
}

function initActiveDocLinkScroll(): void {
  const active = document.querySelector<HTMLElement>(".doc-link.is-active");

  if (!active) {
    return;
  }

  active.scrollIntoView({ block: "center" });
}

async function bootstrap(): Promise<void> {
  initThemeToggle();
  initMenuToggle();
  initSidebarDismiss();
  initDocumentFilter();
  initLandingFilter();
  initCodeCopyButtons();
  initTableOfContentsHighlight();
  initRevealAnimation();
  initMathFallbackRendering();
  initActiveDocLinkScroll();
  await initMermaid();
}

void bootstrap();
