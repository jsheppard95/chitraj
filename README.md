# chitraj

## Installation
```
conda env create -f environment_linux.yml -n chilife_env
conda activate chilife_env
pip install SciencePlots
```

## Usage
Run ChiLife over MD trajectory to obtain DEER distribution for specified sites, averaged over MD trajectory.

### Basic Usage
Compare distance distribution between residues 406 and 537 at 1 bar and 3 kbar. Note these residues are indexed 4 and 135 for 3 kbar. Compare R1 and GTN spin labels using MC sampling over corresponding rotamer libraries.

```
python compare_R1_rotlib_traj.py \
    --top1 top_1bar.pdb \
    --traj1 traj_1bar.xtc \
    --top2 top_3kbar.pdb \
    --traj2 traj_3kbar.xtc \
    --site1-cond1 406 \
    --site2-cond1 537 \
    --site1-cond2 4 \
    --site2-cond2 135 \
    --label-name GTN \
    --sample 5000 \
    --rotlib GTN_rotlib.npz \
    --start-frame 0 \
    --max-frames 30000 \
    --stride 10 \
    --outdir gtn_sample_5000 \
    --tag GTN_sample_5000
```
This creates the directory `gtn_sample_5000` containing per-frame summaries and the average distribution with 95% confidence intervals.

### Trajectory Calculation
Parallelization is achieved by "chunking" trajectory into pieces, with `CHUNK_SIZE` frames each, running each chunk within a SLURM job array.

In `submit_array.sh`, edit `TOP1` and `TRAJ1` to point to the first condition topology and trajectory files, `SITE_COND1` and `SITE_COND2` to the residues of interest, and similarly for the second condition. Also edit `MODE` to select the ChiLife rotamer library and number of MC samples.
```
sbatch submit_array.sh
```

This will create directories `logs` and `chunk_output`. If any chunks fail, run the following bash script to re-run failed chunks. Note this should be run from the command line, not as a SLURM job.

```
./submit_missing_chunks.sh
```

This will inspect `chunk_output` and call `rerun_missing_chunks.sbatch`, again using a job array. Re-run `submit_missing_chunks.sh` as needed.

Combine ChiLife Chunks with the following python script:
```
python combine_chilife_chunks.py \
    --chunk-dir stride_1/gtn_sample_5000/chunk_output
    --prefix GTN_sample_5000
    --outdir stride_1/gtn_sample_5000/combined_output
```

Plot the combined output data, ensuring to edit `MODEL_FILES` to point to the combined output data produced by `combine_chilife_chunks.py`:
```
python plot_R1_rotlib_traj.py
```

### Cluster Calculation
Computes an ensemble-averaged distribution by running ChiLife over RMSD-determined clusters, weighted by their populations. Assumes clustering performed with GROMACS yielding `clusters.pdb` and `cluster.xvg` files.

```
python compare_R1_rotlib_clusters.py \
    --pdb1 clusters_1bar.pdb \
    --xvg1 clust-size_1bar.xvg \
    --pdb2 clusters_3kbar.pdb \
    --xvg2 clust-size_3kbar.xvg \
    --site1-cond1 406 \
    --site2-cond1 537 \
    --site1-cond2 4 \
    --site2-cond2 135 \
    --label-name GTN \
    --sample 5000 \
    --rotlib GTN_rotlib.npz \
    --outdir cluster_output \
    --prefix GTN_sample_5000
```

Plot the cluster results. This is similar in structure to `plot_R1_rotlib_traj.py` but reads cluster-weighted ChiLife results.
```
python plot_cluster_label_comparison.py
```

### Compare with MD
Assumes previously performed ChiLife calculation and GROMACS distance.

Overlay MD distribution with cluster-weighted ChiLife distribution.
```
python plot_cluster_label_md_comparison.py
```

Overlay MD with per-frame ChiLife average distribution.
```
python plot_trajlabel_md_comparison.py
```