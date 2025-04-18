library(data.table)

setwd("/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/")

ctrl<-fread("controls_clumped_genotypes_single_letter.csv")
cases<- fread("cases_clumped_genotypes_single_letter.csv")
cases1<-cases[,7:26]

cc1 <- cbind(ctrl, cases1)
# Make sure cc1 is a data.table
cc1 <- as.data.table(cc1)

# Step 1: Extract sample columns and SNP column
sample_data <- cc1[, 7:46, with = FALSE]
snp_vector <- cc1$SNP  # This is a vector of SNPs, one per row

# Step 2: Transpose so samples become rows
transposed <- transpose(sample_data)

# Step 3: Assign sample names
transposed[, Sample := colnames(sample_data)]

# Step 4: Collapse the base sequence for each sample
transposed[, sequence := apply(.SD, 1, paste, collapse = ""), .SDcols = 1:ncol(sample_data)]


# Step 5: Assign control/case labels
transposed[, case_control := rep(c("control", "case"), each = 20)]

# Step 6: Create SNP string (same for all samples)
snp_concat <- paste(snp_vector, collapse = "; ")

# Step 7: Add SNP string to all rows
transposed[, SNP := snp_concat]

# Step 8: Reorder and select final columns
final_dt <- transposed[, .(Sample, sequence, case_control, SNP)]




write.csv(final_dt, "/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/ayub_clumped_samples_sequences_with_labels_withSNP.csv", row.names=F)


# Keep only relevant columns
final_dt1 <- transposed[, .(Sample, sequence, case_control)]

write.csv(final_dt1, "/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/ayub_clumped_samples_sequences_with_labels.csv", row.names=F)


