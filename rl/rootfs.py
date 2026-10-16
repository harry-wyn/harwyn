"""Unpack a Docker Hub image into a plain directory (no Docker daemon, no root needed).
usage: python rootfs.py REPO TAG OUT_DIR      e.g. xiaomimimo/mimo-v2.6-rl-oss format-code-task-001457 /data/fs/x
Commands then run inside it with proot (see sandbox.py). Handles OCI/Docker manifests and image indexes
(amd64), gzip/zstd layers and whiteouts. Writes OUT_DIR/.image_config.json (env, workdir) and OUT_DIR/.ok.
"""
import io, json, os, shutil, sys, tarfile, urllib.request

def _req(url, tok, accept=None):
    h = {"Authorization": "Bearer " + tok}
    if accept: h["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600)

ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])

def pull(repo, tag, out):
    if os.path.exists(f"{out}/.ok"):
        return
    tok = json.load(urllib.request.urlopen(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull", timeout=60))["token"]
    base = f"https://registry-1.docker.io/v2/{repo}"
    m = json.load(_req(f"{base}/manifests/{tag}", tok, ACCEPT))
    if "manifests" in m:
        d = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"][0]["digest"]
        m = json.load(_req(f"{base}/manifests/{d}", tok, ACCEPT))
    cfg = json.load(_req(f"{base}/blobs/{m['config']['digest']}", tok))
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    for layer in m["layers"]:
        tmp = f"{out}.layer"
        with _req(f"{base}/blobs/{layer['digest']}", tok) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        mode = "r:gz" if "gzip" in layer["mediaType"] else "r:*"
        if "zstd" in layer["mediaType"]:
            import zstandard; fobj = zstandard.ZstdDecompressor().stream_reader(open(tmp, "rb")); tf = tarfile.open(fileobj=fobj, mode="r|")
        else:
            tf = tarfile.open(tmp, mode)
        for ti in tf:
            name = ti.name.lstrip("./")
            d, b = os.path.split(name)
            if b == ".wh..wh..opq":
                p = os.path.join(out, d)
                if os.path.isdir(p):
                    for c in os.listdir(p): shutil.rmtree(os.path.join(p, c), ignore_errors=True) if os.path.isdir(os.path.join(p, c)) and not os.path.islink(os.path.join(p, c)) else os.remove(os.path.join(p, c))
                continue
            if b.startswith(".wh."):
                p = os.path.join(out, d, b[4:])
                if os.path.islink(p) or os.path.isfile(p): os.remove(p)
                elif os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                continue
            if ti.isdev(): continue
            p = os.path.join(out, name)
            if os.path.lexists(p) and not (ti.isdir() and os.path.isdir(p) and not os.path.islink(p)):
                shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
            try:
                tf.extract(ti, out, set_attrs=ti.isdir() is False, filter="tar" if hasattr(tarfile, "tar_filter") else None)
            except Exception:
                pass
        tf.close(); os.remove(tmp)
    json.dump(cfg.get("config", {}), open(f"{out}/.image_config.json", "w"))
    open(f"{out}/.ok", "w").close()

if __name__ == "__main__":
    pull(*sys.argv[1:4]); print("ok", sys.argv[3])
