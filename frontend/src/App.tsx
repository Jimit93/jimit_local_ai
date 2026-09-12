import { useEffect, useState } from "react";
import Benchmarks from "./pages/Benchmarks";
import Chat from "./pages/Chat";
import Knowledge from "./pages/Knowledge";
import Settings from "./pages/Settings";
import System from "./pages/System";
import Tasks from "./pages/Tasks";
import Writer from "./pages/Writer";
import { api } from "./services/api";

const PAGES = {
  chat: { label: "Chat", el: <Chat /> },
  writer: { label: "LinkedIn Writer", el: <Writer /> },
  knowledge: { label: "Knowledge", el: <Knowledge /> },
  tasks: { label: "Tasks", el: <Tasks /> },
  benchmarks: { label: "Benchmarks", el: <Benchmarks /> },
  system: { label: "System", el: <System /> },
  settings: { label: "Settings", el: <Settings /> },
} as const;

type PageKey = keyof typeof PAGES;

export default function App() {
  const [page, setPage] = useState<PageKey>("chat");
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth(null));
    load();
    const t = setInterval(load, 6000);
    return () => clearInterval(t);
  }, []);

  const state = health?.warmup?.state;
  const dot = !health ? "var(--bad)" : state === "ready" ? "var(--ok)" : "var(--warn)";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>Jimit Local AI</h1>
          <small>offline agent workstation</small>
        </div>
        {(Object.keys(PAGES) as PageKey[]).map((k) => (
          <button key={k} className={`nav ${page === k ? "active" : ""}`}
            onClick={() => setPage(k)}>
            {PAGES[k].label}
          </button>
        ))}
        <div className="sidebar-foot">
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: dot }} />
            {!health ? "backend offline" : state === "ready" ? "ready" : `${state || "starting"}…`}
          </div>
          <div className="mono" style={{ marginTop: 6, wordBreak: "break-all" }}>
            {health?.model || ""}
          </div>
          <div style={{ marginTop: 6 }}>
            RAM free {health?.system?.ram_available_gb ?? "—"} GB
          </div>
        </div>
      </aside>
      <main className="main">{PAGES[page].el}</main>
    </div>
  );
}
