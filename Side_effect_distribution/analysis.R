
library(tidyverse)
library(magrittr)
library(janitor)
library(pheatmap)
library(igraph)
library(ggraph)
library(tidygraph)


setwd("/Users/seibi/projects/bmi550/final/side_effect_distribution")



data = read_csv("./results/bc_drug_side_effect_estimated.csv")

data = data %>% 
    dplyr::select(account_id = `4`, drug_name, symptom_id = `Symptom ID`, negation = `Negation flag`) 



# drug distribution
res_list = list()
for (i in 1:nrow(data)){
    #i = 1    
    id = data[i,] %>% pull(account_id) %>% as.character()
    # print(id)
    drugs = unlist(str_split(data[i, "drug_name"], ",", simplify = FALSE))
    drugs = unique(tolower(drugs))
    res_list[[id]] = 
        tibble(
            id = rep(id, length(drugs)),
            drug_used = drugs)
}
res = as_tibble(do.call(rbind, res_list)) 


# load breast cancer approved drugs
drug_dict = read_csv("/Users/seibi/projects/bmi550/final/drug_discovery/breast_cancer_drugs_cleaned.csv")

drug_dict = drug_dict %>% 
    mutate(parent_uspmg = tolower(parent_USPMG),
    parent_nci = tolower(parent_NCI),
    drug = tolower(drug),
    us_brand_name = tolower(us_brand_name)
    ) %>% 
    dplyr::select(parent_uspmg, parent_nci, drug, us_brand_name, functional_class)


# add more information to the discovered drugs that were reported
res_list = list()    
for (i in 1:nrow(res)){
    # i = 1
    id = res[i,] %>% pull(id)
    drug_in_use = res[i, ] %>% pull(drug_used)

    # search in drug or us_brand_name
    res_list[[as.character(i)]] = 
        drug_dict %>% filter(drug == drug_in_use | us_brand_name == drug_in_use) %>% 
                    mutate(id = id,  
                            drug_used = drug_in_use
                            )
}    
res = as_tibble(do.call(rbind, res_list))


# 198 accounts





# drug expressed
n_id = length(unique(res$id))
temp = res %>% 
    filter(!duplicated(cbind(id, drug_used))) %>% 
    group_by(drug_used) %>% 
    summarise(count =n()) %>%
    mutate(prop = round(count/n_id * 100,1)) %>%  
    arrange(desc(count))
p = temp %>% 
    mutate(drug_used_f = factor(drug_used, levels = temp$drug_used)) %>% 
    ggplot(aes(drug_used_f, prop, fill = drug_used_f)) + 
        geom_bar(stat = "identity", show.legend = FALSE) +
        geom_text(aes(label = count)) + 
        theme_bw() + 
        theme(
            axis.text.x = element_text(angle = 30),
            text = element_text(size  = 12)
        ) + 
        xlab("") +
        ylab("Proportion (%) of drugs expressed in Twitter breast cancer posts") 
p
ggsave("./results/bar_drug_expressed.png", p, height = 10, width = 12, unit = "in", dpi = 200)




# authentic drug name
temp = res %>% 
    filter(!duplicated(cbind(id, drug))) %>% 
    group_by(drug) %>% 
    summarise(count =n()) %>%
    mutate(prop = round(count/n_id * 100,1)) %>%  
    arrange(desc(count))
p = temp %>% 
    mutate(drug_f = factor(drug, levels = temp$drug)) %>% 
    ggplot(aes(drug_f, prop, fill = drug_f)) + 
        geom_bar(stat = "identity", show.legend = FALSE) +
        geom_text(aes(label = count)) + 
        theme_bw() + 
        theme(
            axis.text.x = element_text(angle = 30),
            text = element_text(size  = 12)
        ) + 
        xlab("") +
        ylab("Proportion (%) of drug name in Twitter breast cancer posts") 
p
ggsave("./results/bar_drug_name.png", p, height = 10, width = 12, unit = "in", dpi = 200)



# parent_nci
temp = res %>% 
    filter(!duplicated(cbind(id, parent_nci))) %>%
    mutate(parent_nci = ifelse(is.na(parent_nci), "Unknown", parent_nci)) %>% 
    group_by(parent_nci) %>% 
    summarise(count =n()) %>%
    mutate(prop = round(count/n_id * 100,1))  %>% 
    arrange(desc(prop)) 
p = temp %>% 
    mutate(parent_nci_f = factor(parent_nci, levels = temp$parent_nci)) %>% 
    ggplot(aes(parent_nci_f, prop, fill = parent_nci_f)) + 
        geom_bar(stat = "identity", show.legend = FALSE) +
        geom_text(aes(label = count)) + 
        theme_bw() + 
        theme(
            axis.text.x = element_text(angle = 30),
            text = element_text(size  = 12)
        ) + 
        xlab("") +
        ylab("Proportion (%) of parent NCI classification in Twitter breast cancer posts") 
p
ggsave("./results/bar_parent_nci_name.png", p, height = 10, width = 12, unit = "in", dpi = 200)








# functional classification
n_id = length(unique(res$id))
temp = res %>% 
    filter(!duplicated(cbind(id, functional_class))) %>% 
    group_by(functional_class) %>% 
    summarise(count =n()) %>%
    mutate(prop = round(count/n_id * 100,1)) %>%  
    arrange(desc(count))
p = temp %>% 
    mutate(functional_class_f = factor(functional_class, levels = temp$functional_class)) %>% 
    ggplot(aes(functional_class_f, prop, fill = functional_class_f)) + 
        geom_bar(stat = "identity", show.legend = FALSE) +
        geom_text(aes(label = count)) + 
        theme_bw() + 
        theme(
            axis.text.x = element_text(angle = 30),
            text = element_text(size  = 12)
        ) + 
        xlab("") +
        ylab("Proportion (%) of drugs based on the functional classification in Twitter breast cancer posts") 
p
ggsave("./results/bar_functional.png", p, height = 10, width = 12, unit = "in", dpi = 200)















# save
drug_res = res











# side effect reported distribution
res_list = list()
for (i in 1:nrow(data)){
    # i = 20    
    id = data[i,] %>% pull(account_id) %>% as.character()
    
    sym_id = unlist(str_split(data[i, "symptom_id"], regex("\\$\\$\\$"), n = Inf, simplify = FALSE))
    # remove empty
    sym_id = sym_id[!sym_id == ""]


    negs = unlist(str_split(data[i, "negation"], regex("\\$\\$\\$"), n = Inf, simplify = FALSE))
    # remove empty
    negs = negs[!negs == ""]

    # remove negated symptoms
    negated_idx = negs == "1"

    # positive symptom
    sym_id = sym_id[!negated_idx]

    # remove dups
    sym_id = sym_id[!duplicated(sym_id)]

    res_list[[id]] = 
        tibble(
            id = rep(id, length(sym_id)),
            sym_id_used = sym_id)
}
res = as_tibble(do.call(rbind, res_list)) 



# load symptom dictionary
sym_dict = read_csv("/Users/seibi/projects/bmi550/final/drug_discovery/covid_bc_sideeffect_dictionary_ver3.csv") %>% 
    distinct(category, id) %>% 
    dplyr::rename(sym_id = id)


res = res %>%
    left_join(sym_dict, by = c("sym_id_used" = "sym_id")) 





# distibution of posotive symptoms 
temp = res %>% 
    filter(!duplicated(cbind(id, category))) %>% 
    group_by(category) %>% 
    summarise(count =n()) %>%
    mutate(prop = round(count/n_id * 100,1)) %>%  
    arrange(desc(prop))
p = temp %>% 
    mutate(category_f = factor(category, levels = temp$category)) %>% 
    ggplot(aes(category_f, prop, fill = category_f)) + 
        geom_bar(stat = "identity", show.legend = FALSE) +
        geom_text(aes(label = count)) + 
        theme_bw() + 
        theme(
            axis.text.x = element_text(angle = 30),
            text = element_text(size  = 12)
        ) + 
        xlab("") +
        ylab("Proportion (%) of symptoms reported in Twitter breast cancer posts") 
p
ggsave("./results/bar_side_effect_expressed.png", p, height = 10, width = 12, unit = "in", dpi = 200)



# save
side_effect_res = res












# interaction of drug and side effect
temp = drug_res %>% left_join(side_effect_res, by = "id") %>%
    mutate(category = ifelse(is.na(category), "No reported symps", category)) %>%     
    group_by(drug, category) %>% 
    summarise(count = n()) %>% 
    ungroup()  


# should normalize by the total number of symptoms per drug
drug_count_dict = temp %>% 
    group_by(drug) %>% 
    summarise(drug_count = sum(count)) %>%
    ungroup()


temp1 = temp %>% 
    left_join(drug_count_dict, by = "drug") %>% 
    mutate(normalized_count = count/drug_count) %>% 
    dplyr::select(drug, category, normalized_count) %>% 
    pivot_wider(names_from = "drug", values_from = "normalized_count", values_fill = 0) 


mat = temp1[,2:ncol(temp1)] %>% as.matrix()
rownames(mat) = temp1$category


# add functional class
drug_dict = drug_res %>% distinct(drug, functional_class) 
anno = tibble(drug = colnames(mat)) %>% 
    left_join(drug_dict, by ="drug") %>% 
    dplyr::select(functional_class) %>% as.data.frame()
rownames(anno) = colnames(mat)



paletteLength = 50
myColor = colorRampPalette(c("white", "#ff2600"))(paletteLength)

p = pheatmap(mat, color = myColor, annotation_col = anno)
ggsave("./results/heat_drug_symptom.png", p, height = 10, width = 12, unit = "in", dpi = 200)











# network
link = temp %>% 
    left_join(drug_count_dict, by = "drug") %>% 
    mutate(normalized_count = count/drug_count) %>% 
    dplyr::select(drug, category, normalized_count) 

nodes = tibble(
    name = c(unique(link$drug), unique(link$category)),
    type = c(rep("Drug", length(unique(link$drug))), rep("Symptoms", length(unique(link$category))))
)    


network = graph_from_data_frame(link, nodes, directed = TRUE) 

p = ggraph(network, layout = "circle") +
    geom_edge_link(aes(edge_alpha = normalized_count, width = normalized_count, color = normalized_count)) + 
    scale_edge_color_continuous(name = "Normalized counts") +
    scale_edge_width_continuous(name = "Normalized counts") +
    scale_edge_alpha_continuous(name = "Normalized counts") +
    geom_node_point(aes(fill = type), shape = 21, size= 10, alpha = 1, color = "black") +
    geom_node_text(aes(label = name), repel = TRUE) + 
    scale_fill_manual(name = "Type", values = c("palegreen", "#e1a9f4d8")) +
    theme_bw()+
    xlab("") +
    ylab("")+
    theme(
            legend.text = element_text(size =12),
            strip.text = element_text(size = 18)
        )

p
ggsave("./results/network.png", p, height = 10, width = 12, unit = "in", dpi = 200)    
















# interaction of functional classification and side effect
temp = drug_res %>% left_join(side_effect_res, by = "id") %>%
    mutate(category = ifelse(is.na(category), "No reported symps", category)) %>%     
    group_by(functional_class, category) %>% 
    summarise(count = n()) %>% 
    ungroup()  


# should normalize by the total number of symptoms per drug
drug_count_dict = temp %>% 
    group_by(functional_class) %>% 
    summarise(drug_count = sum(count)) %>%
    ungroup()


temp1 = temp %>% 
    left_join(drug_count_dict, by = "functional_class") %>% 
    mutate(normalized_count = count/drug_count) %>% 
    dplyr::select(functional_class, category, normalized_count) %>% 
    pivot_wider(names_from = "functional_class", values_from = "normalized_count", values_fill = 0) 
mat = temp1[,2:ncol(temp1)] %>% as.matrix()
rownames(mat) = temp1$category



numbers = temp %>% 
    left_join(drug_count_dict, by = "functional_class") %>% 
    mutate(normalized_count = count/drug_count) %>% 
    dplyr::select(functional_class, category, count) %>% 
    pivot_wider(names_from = "functional_class", values_from = "count", values_fill = 0) 
num_mat = numbers[,2:ncol(numbers)] %>% as.matrix()
rownames(num_mat) = numbers$category


paletteLength = 50
myColor = colorRampPalette(c("white", "#ff2600"))(paletteLength)

p = pheatmap(mat, color = myColor, display_numbers = num_mat)

ggsave("./results/heat_functional_class_symptom.png", p, height = 10, width = 12, unit = "in", dpi = 200)




# proportion t-test

aov_symptoms = function(sym){
    #sym = "Pyrexia"
    anova_data = temp %>%
        pivot_wider(names_from = "category", values_from ="count", values_fill = 0) %>% 
        pivot_longer(!functional_class, names_to = "category", values_to = "count") %>% 
        left_join(drug_count_dict, by = "functional_class") %>% 
        mutate(normalized_count = count/drug_count) %>% 
        filter(category == sym)


    chemo_total_posts = anova_data %>% filter(functional_class=="Chemotherapy") %>% pull(drug_count)
    hor_total_posts = anova_data %>% filter(functional_class=="Hormone therapy") %>% pull(drug_count)
    immune_total_posts = anova_data %>% filter(functional_class=="Immune exhaustion inhibitor") %>% pull(drug_count)
    kinase_total_posts = anova_data %>% filter(functional_class=="Kinase inhibitor") %>% pull(drug_count)

    chemo_sym_posts = anova_data %>% filter(functional_class=="Chemotherapy") %>% pull(count)
    hor_sym_posts = anova_data %>% filter(functional_class=="Hormone therapy") %>% pull(count)
    immune_sym_posts = anova_data %>% filter(functional_class=="Immune exhaustion inhibitor") %>% pull(count)
    kinase_sym_posts = anova_data %>% filter(functional_class=="Kinase inhibitor") %>% pull(count)


    anova_data_temp = tibble(
        functional = factor(c( rep("chemo", chemo_total_posts), rep("hormone", hor_total_posts),
                        rep("immu", immune_total_posts), rep("kinase", kinase_total_posts))), 
        sym_yes = c(rep(1, chemo_sym_posts),  rep(0,  chemo_total_posts - chemo_sym_posts),
                    rep(1, hor_sym_posts),    rep(0,  hor_total_posts - hor_sym_posts),
                    rep(1, immune_sym_posts),  rep(0, immune_total_posts - immune_sym_posts),
                    rep(1, kinase_sym_posts),  rep(0, kinase_total_posts - kinase_sym_posts))
        )


    fit = aov(sym_yes ~functional, data = anova_data_temp)
    pval = anova(fit)$"Pr(>F)"[1] 
    return(pval)
}


anova_list = list()
for( i in unique(temp$category)){
    anova_list[[i]] = tibble(
        symptom = i,
        pval =aov_symptoms(i) 
        )
}

# no symptom with padj <0.05
as_tibble(do.call(rbind, anova_list)) %>% 
    mutate(padj = p.adjust(pval, method = "BH", n =nrow(as_tibble(do.call(rbind, anova_list))) )) 

sig_syms = as_tibble(do.call(rbind, anova_list))  %>% filter(pval < 0.05) %>% 
    pull(symptom)


# symptom                               pval
# <chr>                                <dbl>
# Nerve Problems                     0.00377
# No reported symps                  0.00325
# Cough                              0.00714
# Dizziness/disorientation/confusion 0.00232


# boxplot
plots = list()
for(sym in sig_syms){
    p = temp %>%
        pivot_wider(names_from = "category", values_from ="count", values_fill = 0) %>% 
        pivot_longer(!functional_class, names_to = "category", values_to = "count") %>% 
        left_join(drug_count_dict, by = "functional_class") %>% 
        mutate(normalized_count = count/drug_count) %>%
        rowwise() %>% 
        mutate(label_ = paste0(count, "/",drug_count)) %>% 
        ungroup() %>% 
        filter(category == sym) %>% 
        ggplot(aes(functional_class, normalized_count, fill = functional_class)) + 
            geom_bar(stat = "identity") + 
            geom_text(aes(label = label_)) +
            theme_bw() + 
            theme(
                text = element_text(size = 15)
            ) +
            xlab("") + 
            ylab("Normalized counts of side effect") +
            ggtitle(sym) +
            scale_fill_discrete(name = "Functional class")
    plots[[sym]] = p
}
p = egg::ggarrange(plots = plots, ncol = 1)

ggsave("./results/sig_symps.png", p, height = 14, width = 10, unit = "in", dpi = 200)















