export interface Heading {
  level: number;
  id: string;
  text: string;
}

export interface RenderEnv {
  headings: Heading[];
}

export interface DocumentPage {
  sourcePath: string;
  relativeMarkdownPath: string;
  outputRelPath: string;
  title: string;
  html: string;
  headings: Heading[];
}
