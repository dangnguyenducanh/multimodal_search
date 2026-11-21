import { useEffect, useState } from "react";
import type { ApiErrorBody } from "@/types/api";

export class ApiError extends Error {
  status: number;
  code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function isApiErrorBody(x: unknown): x is ApiErrorBody {
  return typeof x === "object" && x !== null && "error" in x;
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal });
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    if (isApiErrorBody(body)) throw new ApiError(res.status, body.error.code, body.error.message);
    throw new ApiError(res.status, "HTTP_ERROR", `request failed (${res.status})`);
  }
  return body as T;
}

/** {data, error, loading} for a GET; a new url aborts the previous request. null url = idle. */
export function useApi<T>(url: string | null) {
  const [state, setState] = useState<{ data: T | null; error: ApiError | null; loading: boolean }>({
    data: null,
    error: null,
    loading: url !== null,
  });
  useEffect(() => {
    if (url === null) return;
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    getJson<T>(url, ctrl.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        const err = e instanceof ApiError ? e : new ApiError(0, "NETWORK", "network error");
        setState({ data: null, error: err, loading: false });
      });
    return () => ctrl.abort();
  }, [url]);
  return state;
}
