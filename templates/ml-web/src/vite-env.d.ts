/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the FastAPI service. Only the API URL is exposed to the browser; no keys belong here. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
