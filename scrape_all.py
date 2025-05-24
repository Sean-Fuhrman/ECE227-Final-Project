#%%
import ctypes, numpy as np, pandas as pd, networkit as nk
#%%
###############################################################################
# 0)  One alias so NetworKit 10.1 and NumPy ≥2.0 cooperate
###############################################################################
if not hasattr(np, "ulong"):
    # match the *native* width of `unsigned long` on the current OS
    np.ulong = np.uint64 if ctypes.sizeof(ctypes.c_ulong) == 8 else np.uint32

###############################################################################
# 1)  Pre-allocate roughly the final number of nodes (60 M for English Wiki)
###############################################################################
G = nk.graph.Graph(60_000_000, weighted=False, directed=True)

###############################################################################
# 2)  Stream edges in chunks and append titles to a Parquet file
###############################################################################
CHUNK  = 1_000_000
TITLE_FILE = "node_titles.parquet"
parquet_writer = None  # will be opened on first chunk

reader = pd.read_csv(
    "data/enwiki.wikilink_graph.2018-03-01.csv.gz",
    sep="\t",
    header=0,
    names=["u", "u_title", "v", "v_title"],
    dtype={"u": np.ulong, "v": np.ulong},   # correct width for NetworKit
    chunksize=CHUNK,
    compression="gzip",
    engine="c",
    on_bad_lines="skip",
)

import pyarrow as pa, pyarrow.parquet as pq

for chunk in reader:
    # -- 2a) store titles ----------------------------------------------------
    titles = (
        pd.concat(
            [
                chunk[["u", "u_title"]].rename(columns={"u": "id", "u_title": "title"}),
                chunk[["v", "v_title"]].rename(columns={"v": "id", "v_title": "title"}),
            ],
            ignore_index=True,
        )
        .drop_duplicates("id")
    )
    print(titles)
    tbl = pa.Table.from_pandas(titles, preserve_index=False)

    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(TITLE_FILE, tbl.schema, compression="zstd")
    parquet_writer.write_table(tbl)

    # -- 2b) add edges -------------------------------------------------------
    src = chunk["u"].to_numpy(dtype=np.ulong, copy=False)
    dst = chunk["v"].to_numpy(dtype=np.ulong, copy=False)

    # single *inputData* argument: a tuple of the two ndarrays
    G.addEdges((src, dst), addMissing=False, checkMultiEdge=False)

if parquet_writer is not None:
    parquet_writer.close()

nk.graphio.writeGraph(G, "enwiki_2018.nkbg", nk.Format.NetworkitBinary)

#%%

import gc, networkit as nk

G = nk.graphio.readGraph("data/enwiki_2018.nkbg", nk.Format.NetworkitBinary)

sinks = [u for u in G.iterNodes() if G.degree(u) == 0]
print(f"• Removing {len(sinks):,} sink nodes…")
for u in sinks:
    G.removeNode(u)  # <- exists in all NetworKit versions

nk.graphio.writeGraph(G, "data/enwiki_2018_core.nkbg",
                      nk.Format.NetworkitBinary)

print(f"✓  Saved pruned graph:"
      f" {G.numberOfNodes():,} nodes / {G.numberOfEdges():,} edges")


#%%