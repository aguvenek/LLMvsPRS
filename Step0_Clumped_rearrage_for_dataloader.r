library(data.table)

# ---- Parameters ----
setwd("/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/")
dataset_name <- "ayub"  # <- Set this per dataset

# ---- Load data ----
ctrl <- fread("controls_clumped_genotypes_single_letter.csv")
cases <- fread("cases_clumped_genotypes_single_letter.csv")

# Extract genotype columns (assuming from column 7 onward)
ctrl1 <- ctrl[, 7:ncol(ctrl), with = FALSE]
cases1 <- cases[, 7:ncol(cases), with = FALSE]

# Combine sample matrices
all_samples <- cbind(ctrl1, cases1)

# Get sample names and case/control labels
sample_names <- c(colnames(ctrl1), colnames(cases1))
labels <- c(rep(0, ncol(ctrl1)), rep(1, ncol(cases1)))

# Create full-length sequences efficiently
sequences <- vapply(seq_along(sample_names), function(i) {
  paste0(all_samples[[i]], collapse = "")
}, character(1))

# ---- Create base result table ----
result <- data.table(
  sample = sample_names,
  sequence = sequences,
  labels = labels,
  dataset = dataset_name
)

# ---- Save version without SNPs ----
fwrite(result, "../ayub_clumped_samples_sequences_with_labels.csv", row.names = FALSE)

# ---- Create and save version with SNPs ----
snp_vector <- ctrl$SNP  # assumes SNP column is present in ctrl
snp_concat <- paste(snp_vector, collapse = "; ")
result_with_snp <- copy(result)
result_with_snp[, SNP := snp_concat]
fwrite(result_with_snp, "../ayub_clumped_samples_sequences_with_labels_withSNP.csv", row.names = FALSE)