declare module "markdown-it-task-lists" {
  import type MarkdownIt from "markdown-it";

  interface TaskListOptions {
    enabled?: boolean;
    label?: boolean;
    labelAfter?: boolean;
  }

  const taskLists: (md: MarkdownIt, options?: TaskListOptions) => void;
  export default taskLists;
}

declare module "markdown-it-texmath" {
  import type MarkdownIt from "markdown-it";

  interface TexMathOptions {
    engine: unknown;
    delimiters?: "dollars" | "brackets" | "kramdown" | "gitlab" | "julia";
    katexOptions?: Record<string, unknown>;
    outerSpace?: boolean;
  }

  const texmath: (md: MarkdownIt, options: TexMathOptions) => void;
  export default texmath;
}

declare module "katex/contrib/auto-render" {
  interface Delimiter {
    left: string;
    right: string;
    display: boolean;
  }

  interface RenderMathInElementOptions {
    delimiters?: Delimiter[];
    ignoredTags?: string[];
    ignoredClasses?: string[];
    throwOnError?: boolean;
    strict?: "warn" | "ignore" | "error" | boolean;
  }

  export default function renderMathInElement(
    element: HTMLElement,
    options?: RenderMathInElementOptions
  ): void;
}

declare module "markdown-it-emoji" {
  import type MarkdownIt from "markdown-it";

  interface EmojiModule {
    (md: MarkdownIt): void;
    full?: (md: MarkdownIt) => void;
    light?: (md: MarkdownIt) => void;
    bare?: (md: MarkdownIt) => void;
    default?: (md: MarkdownIt) => void;
  }

  const emoji: EmojiModule;
  export = emoji;
}
