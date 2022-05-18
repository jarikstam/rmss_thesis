# Setup ----
library(shiny)
library(tidyverse)
library(bslib)

x <- comments %>%
  select(ends_with("_cmd") | ends_with("_centroid") | film_title | Date | tokenized_body) %>% 
  mutate(grp = cut_interval(discrimination_centroid, length = .5)) %>% 
  group_by(grp) %>% 
  slice_sample(n=5) %>% 
  ungroup() %>% 
  select(!grp)

cmd_vars <- comments %>% select(ends_with("_cmd") | ends_with("_centroid")) %>% colnames()

# UI ----
ui <- fluidPage(
  theme = bs_theme(bootswatch = "flatly"),
  fluidRow(
    column(3, selectInput("var", "CMD Variable", cmd_vars, selected = "discrimination_centroid")),
    # column(3, numericInput("min", "Min Score", value = 5, step = .5)),
    # column(3, numericInput("max", "Max Score", value = 13, step = .5))
    column(6, sliderInput("score", label = "Standardized Concept Mover's Distance", min = -4.5, max = 11, value = c(5, 11), step = .5, width="100%"))
    # column(3, numericInput("min_comment_size", "Min. Comment Size", value = 0, step = 1)),
    # column(3, numericInput("max_comment_size", "Max. Comment Size", value = 1000, step = 1))
    ),
  fluidRow(column(12, tableOutput("comment_info")))
)
# Server ----
server <- function(input, output, session) {
  output$comment_info <- renderTable(
    x %>%
      filter(.data[[input$var]] >= input$score[1],
             .data[[input$var]] <= input$score[2],
             # str_count(tokenized_body, "\\w+") >= input$min_comment_size,
             # str_count(tokenized_body, "\\w+") <= input$max_comment_size
             ) %>% 
      select(
        tokenized_body, film_title, Date, input$var,  
      ) %>% 
      mutate(Date = as.character(Date)) %>% 
      slice_sample(n=5)
  )
}

shinyApp(ui, server)