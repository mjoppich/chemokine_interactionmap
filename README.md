# Interactionmaps (currently for chemokines only)

Derive a chemokine-chemokine interaction map from (sc)RNA-seq sequencing data

The source code for deriving the required input files is shown below.

    makesummary_getPop = function(a, suffix)
    {
    out = {}
    out["num"] = length(a)
    
    if (length(a) == 0)
    {
        f = c(0,0,0,0,0)
        meanA = 0
    } else {
        f = fivenum(a)
        meanA = mean(a)
    }

    out["min"] = f[1]
    out["lower_hinge"] = f[2]
    out["median"] = f[3]
    out["upper_hinge"] = f[4]
    out["max"] = f[5]
    out["mean"] = meanA
    
    names(out) = paste(names(out), suffix, sep=".")
    
    return(out)
    }

    getExprData_getPop = function(markerObj, markerCells, sampleSuffix, slot="data", assay="RNA")
    {
    expTable = GetAssayData(object = subset(x=markerObj, cells=markerCells), slot = slot, assay=assay)
    allgenes = rownames(expTable)
    cellnames = colnames(expTable)

    expt.r = as(expTable, "dgTMatrix")
    expt.df = data.frame(r = expt.r@i + 1, c = expt.r@j + 1, x = expt.r@x)

    DT <- data.table(expt.df)
    res = DT[, as.list(makesummary_getPop(x, sampleSuffix)), by = r]
    anumCol = paste("anum", sampleSuffix, sep=".")
    res[[anumCol]] = length(cellnames)
    res$gene = allgenes[res$r]

    res = res[,r:=NULL]

    return(res)
    }

    get_population_expression_data = function(scobj, group, outname, assay="RNA")
    {

    exprData = list()

    for (cellPop in unique(as.character(unlist(scobj[[group]]))))
    {
        varPop = str_to_lower( str_replace_all(
                                str_replace_all(#
                                str_replace_all( cellPop, "\\(|\\)| |,", "_"),
                                "__", "_"),
                                "_$", "")
                            )
        print(paste(cellPop, varPop))
        
        allPopCells = scobj[[group]]
        allPopCells$cnames = rownames(allPopCells)
        cellPopCells = allPopCells[allPopCells[[group]] == cellPop, ]$cnames
        print(paste("Number of cells: ", length(cellPopCells)))

        exprData[[varPop]] = getExprData_getPop(markerObj=scobj, markerCells=cellPopCells, sampleSuffix=varPop, slot="counts", assay=assay)
    }


    meanExprData = list()

    for (name in names(exprData))
    {
        
        exprDf = as.data.frame(exprData[[name]])
        subdf = exprDf[ ,c("gene", paste("mean", name, sep=".")) ]

        meanExprData[[name]] = subdf
    }

    cellnames_manualExprDF = Reduce(function(x,y) merge(x = x, y = y, by = "gene", all.x=T, all.y=T), meanExprData)
    cellnames_manualExprDF[is.na(cellnames_manualExprDF)] = 0

    write.table(cellnames_manualExprDF, file = paste(outname, ".tsv", sep=""), quote=FALSE, sep = "\t", row.names = F)
    write_xlsx( cellnames_manualExprDF, path = paste(outname, ".xlsx", sep="") )

    return(cellnames_manualExprDF)


    }

    library("writexl")
    library(stringr)
    library(data.table)

The actual call for deriving cluster-specific expression values is then

    cluster_mean_expr_hybrid = get_population_expression_data(hybridLib, group="idents", outname="cluster_mean_expr_hybrid")

Where hybridLib is the Seurat object, idents is a slot in the object to derive the expression values for, and outname specifies the path and file-prefix (excluding file-type extension) for the tables.

# Methods

Chemokine ligand-chemokine receptor interactions were collected from two different resources:

1) Bhusal, R. P., Eaton, J. R. O., Chowdhury, S. T., Power, C. A., Proudfoot, A. E. I., Stone, M. J., & Bhattacharya, S. (2020). Evasins: Tick Salivary Proteins that Inhibit Mammalian Chemokines. In Trends in Biochemical Sciences (Vol. 45, Issue 2, pp. 108–122). Elsevier Ltd. https://doi.org/10.1016/j.tibs.2019.10.003

2) https://www.rndsystems.com/pathways/chemokine-superfamily-pathway-human-mouse-lig-recept-interactions

Interactions are either classified as antagonist, agonist or undefined.

In general, the steps described by Armingol et al. are followed for determining cell-cell communications.

Armingol, E., Officer, A., Harismendy, O., & Lewis, N. E. (2021). Deciphering cell–cell interactions and communication from gene expression. Nature Reviews Genetics, 22(2), 71–88. https://doi.org/10.1038/s41576-020-00292-x

The experimental expression data (for each cluster) is read in and filtered to only contain genes from the above collection of chemokine interactors.
For each ligand-receptor pair, and for each cluster-pair, the communication score is calculated. This communication score is the product of the ligand expression and the receptor expression (expression product).
This results in a data frame in which for each ligand-receptor pair in each cluster pair a score is associated.

In order to determine the total communication between two clusters, all communication scores between these clusters are aggregated (sum).
In a second step, the data frame is arranged into matrix form, keeping only those clusters of interest (or all, if no filtering was requested).
For highlighting specific interactions (e.g. CCL2->CCR2), the maximal interaction score among all cluster interactions for this specific interaction is determined.
This value is then used to scale the single cluster interactions by the maximally observed interaction score (that is: the ratio of the interaction score divided by the maximal interaction score seen).

These information is then used to plot a chord diagram (taken from https://github.com/tfardet/mpl_chord_diagram) showing LR-interactions between clusters.

In addition, the matrix plot shows the scaled (z-score) expression scores for all interactions in the selected clusters. In a filtered version, only interaction which have at least in one cluster pair a z-score > 1 are shown.

Finally, the chemokines overview displays the ligand-receptor map in the lower left corner, and shows the expression values for the receptors in the selected clusters on top, and those for the ligands to the right.
This visualization allows for a brief overview of ligand and receptor expressions, while also showing possible interactions.

