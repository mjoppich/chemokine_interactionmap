# chemokine_interactionmap
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

    getExprData_getPop = function(markerObj, markerCells, sampleSuffix, slot="data")
    {
    expTable = GetAssayData(object = subset(x=markerObj, cells=markerCells), slot = slot)
    allgenes = rownames(expTable)
    cellnames = colnames(expTable)

    expt.r = as(expTable, "dgTMatrix")
    expt.df = data.frame(r = expt.r@i + 1, c = expt.r@j + 1, x = expt.r@x)

    DT <- data.table(expt.df)
    res = DT[, as.list(makesummary_getPop(x, sampleSuffix)), by = r]
    res[[paste("anum", sampleSuffix, sep=".")]] = length(cellnames)
    res$gene = allgenes[res$r]
    
    res = res[,r:=NULL]
    
    return(res)
    }

    get_population_expression_data = function(scobj, group, outname)
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
    
        exprData[[varPop]] = getExprData_getPop(markerObj=scobj, markerCells=cellPopCells, sampleSuffix=varPop, slot="counts")
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

Where hybridLib is the Seurat object, idents is a slot in the object to dervice the expression values for and outname specifies the path and file-prefix (excluding file-type extension) for the tables.

