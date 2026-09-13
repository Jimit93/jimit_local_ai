import { useState } from "react";
import { Card, Err, Spinner } from "../components/Bits";
import { DiffView } from "../components/AgentBits";
import { agentApi, usePoll, useWorkspace } from "../services/agentApi";

interface TreeNode { name: string; path: string; file?: boolean; children?: TreeNode[] }

function Tree({ node, depth, onOpen, marks, selected }: {
  node: TreeNode; depth: number; onOpen: (p: string) => void; marks: Record<string, string>; selected: string | null;
}) {
  const [open, setOpen] = useState(depth < 1);
  if (node.file) {
    const mark = marks[node.path];
    return (
      <div className={`tree-row file ${selected === node.path ? "sel" : ""}`} style={{ paddingLeft: depth * 14 + 18 }}
        onClick={() => onOpen(node.path)} role="button" title={node.path}>
        {node.name}{mark && <span className={`tree-mark ${mark}`}>{mark === "modified" ? "M" : mark === "relevant" ? "R" : "•"}</span>}
      </div>
    );
  }
  return (
    <div>
      {depth >= 0 && node.path && (
        <div className="tree-row dir" style={{ paddingLeft: depth * 14 + 4 }} onClick={() => setOpen(!open)} role="button"
          aria-expanded={open}>
          {open ? "▾" : "▸"} {node.name}
        </div>
      )}
      {(open || !node.path) && node.children?.map((c) => (
        <Tree key={c.path} node={c} depth={node.path ? depth + 1 : depth} onOpen={onOpen} marks={marks} selected={selected} />
      ))}
    </div>
  );
}

export default function Workspace() {
  const [workspace, setWorkspace] = useWorkspace();
  const [newPath, setNewPath] = useState("");
  const [file, setFile] = useState<any>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [diff, setDiff] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const wsList = usePoll(() => agentApi.workspaces(), 0);
  const tree = usePoll(() => agentApi.tree(workspace || undefined), 0, [workspace]);
  const activity = usePoll(() => agentApi.activity(workspace || undefined), 5000, [workspace]);

  const marks: Record<string, string> = {};
  (activity.data?.recent || []).forEach((r: any) => { marks[r.path] = "recent"; });
  (activity.data?.relevant || []).forEach((r: any) => { marks[r.path] = "relevant"; });
  (activity.data?.modified || []).forEach((r: any) => { marks[r.path] = "modified"; });

  async function open(path: string) {
    setSelected(path);
    setErr(null);
    setDiff(null);
    try {
      setFile(await agentApi.read(path, workspace || undefined));
    } catch (e) {
      setFile(null);
      setErr(String(e));
    }
  }

  async function addWorkspace() {
    setErr(null);
    try {
      const r = await agentApi.addWorkspace(newPath.trim());
      setWorkspace(r.registered);
      setNewPath("");
      wsList.setData({ workspaces: r.workspaces });
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <>
      <div className="page-head">
        <h2>Workspace</h2>
        <p>The agent can only see and change files inside a registered workspace. Secret files are never opened.</p>
      </div>
      <Card title="Workspaces">
        <div className="btn-row">
          <select aria-label="Workspace" value={workspace} onChange={(e) => setWorkspace(e.target.value)} style={{ maxWidth: 420 }}>
            {(wsList.data?.workspaces || []).map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
          <input aria-label="New workspace path" placeholder="C:\\path\\to\\project" value={newPath}
            onChange={(e) => setNewPath(e.target.value)} style={{ maxWidth: 320 }} />
          <button className="btn ghost" disabled={!newPath.trim()} onClick={addWorkspace}>Register</button>
          <button className="btn ghost" onClick={async () => {
            try { const r = await agentApi.index(workspace); alert(`Indexed ${r.files} files (${r.changed} changed) in ${r.ms} ms`); }
            catch (e) { setErr(String(e)); }
          }}>Re-index</button>
        </div>
        {err && <div style={{ marginTop: 10 }}><Err>{err}</Err></div>}
      </Card>

      <div className="ws-grid">
        <Card title={`Project tree${tree.data ? ` · ${tree.data.files} files` : ""}`}>
          {!tree.data && !tree.error && <Spinner label="loading tree" />}
          {tree.error && <Err>{tree.error}</Err>}
          <div className="tree">
            {tree.data && <Tree node={tree.data.root} depth={0} onOpen={open} marks={marks} selected={selected} />}
          </div>
          <div className="muted" style={{ fontSize: 11, marginTop: 8 }}>M modified · R relevant to the last task · • recently accessed</div>
        </Card>

        <div style={{ minWidth: 0 }}>
          <Card title={file ? `${file.path} · ${file.total_lines} lines${file.cached ? " · cached" : ""}` : "File"}
            right={selected && (marks[selected] === "modified") && (
              <button className="btn ghost chip" onClick={async () => setDiff((await agentApi.gitDiff(workspace, selected)).diff)}>
                Show diff</button>)}>
            {diff !== null ? <DiffView diff={diff} /> : file ? (
              <pre className="code mono">{file.content.split("\n").map((l: string, i: number) => (
                <div key={i}><span className="ln">{i + file.start_line}</span>{l || " "}</div>))}</pre>
            ) : <div className="muted">Select a file in the tree.</div>}
            {file?.redactions > 0 && <div className="muted" style={{ marginTop: 6 }}>{file.redactions} secret value(s) redacted.</div>}
          </Card>

          <div className="grid-2">
            <Card title="Modified (git)">
              {(activity.data?.modified || []).slice(0, 25).map((f: any) => (
                <div key={f.path} className="list-row" role="button" onClick={() => open(f.path)}>
                  <span className="pill warn">{f.status}</span> <span className="mono">{f.path}</span></div>
              ))}
              {!activity.data?.modified?.length && <div className="muted">Clean.</div>}
            </Card>
            <Card title="Relevant · recent">
              {(activity.data?.relevant || []).slice(0, 10).map((f: any) => (
                <div key={"r" + f.path} className="list-row" role="button" onClick={() => open(f.path)}>
                  <span className="pill nlp">R</span> <span className="mono">{f.path}</span></div>
              ))}
              {(activity.data?.recent || []).slice(0, 10).map((f: any) => (
                <div key={"a" + f.path} className="list-row" role="button" onClick={() => open(f.path)}>
                  <span className="pill deterministic">{f.action}</span> <span className="mono">{f.path}</span></div>
              ))}
              {!activity.data?.relevant?.length && !activity.data?.recent?.length && <div className="muted">Nothing yet.</div>}
            </Card>
          </div>
        </div>
      </div>
    </>
  );
}
