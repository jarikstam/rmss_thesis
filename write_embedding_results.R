library(tidyverse)

emb_results <- read_csv("data/embedding_results.csv")

txt <- emb_results %>% 
  select(!Wordsim) %>% 
  rename("Wordsim" = "Sp_corr") %>%
  xtable::xtable(
    caption = "Word Embedding Model Comparison",
    label = "Embedding",
    align = rep("C", ncol(.)+1),
    digits = c(0,0,0,3,3)
  ) %>% 
  print(
    table.placement = "tb",
    caption.placement = "top",
    latex.environments = "threeparttable",
    tabular.environment = "tabulary",
    include.rownames = F,
    #math.style.negative = T,
    print.results = F,
    booktabs = T,
    width = "\\linewidth",
    comment = F
  ) %>% 
  str_replace(., "Google", "\\\\% Correct Google Analogies\\\\tnote{a}") %>% 
  str_replace(., "Wordsim", "WordSim-353 Spearman's $\\\\rho$\\\\tnote{b}") %>% 
  str_replace(., "25 & 300 & 0.459 & 0.568", "\\\\textbf{25} & \\\\textbf{300} & \\\\textbf{0.459} & \\\\textbf{0.568}") %>% 
  str_replace(., "end\\{tabulary\\}", "end{tabulary}\n\\\\begin{tablenotes}\n\\\\item \\\\textit{Note.} Model trained using Gensim's Word2Vec. Chosen model in bold.\n\\\\item [a] Google Analogy Test \\\\parencite{Mikolov2013}.\n\\\\item [b] WordSimilarity-353 \\\\parencite{Finkelstein2001}.\n\\\\end{tablenotes}")

# txt
write(txt, "output/embeddingResults.tex")
