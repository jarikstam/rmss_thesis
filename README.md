# The Effect of Social Movements on Online Film Discourse

Description..

## Scripts

### 1. Collecting Reddit Submissions

### 2. Collecting Reddit Comments

## Data Files

### Reddit Film Discussion Comments

[Film\_discussions](data/film\_discussions) is a SQLite database with data collected from film discussion threads on [Reddit.com/r/movies](https://www.reddit.com/r/movies). It contains three tables:
- submissions; collected in [step 1](#1.-collecting-reddit-submissions) which contains the columns:
  - submission\_id
  - title
  - score
  - num\_comments
  - url
  - created
- comments; collected in [step 2](#2.-collecting-reddit-comments) which contains the columns:
  - comment\_id
  - submission\_id
  - body
  - author
  - score
  - created
- topic\_scores; created in [4c\_exploring\_topics.ipynb](4c\_exploring\_topics.ipynb) and contains topic loadings for each comment. Columns are:
  - comment\_id
  - a \(named\) column for each topic

### Twitter Hashtag Frequency

[Keywords\_daily\_count.csv](data/keywords\_daily\_count.csv) contains daily frequencies for several keywords on [Twitter](https://twitter.com). This data was collected in [1d\_collect\_twitter\_data.ipynb](1d\_collect\_twitter\_data.ipynb) and contains the following columns:
- Date
- MeToo
- BLM
- OscarsSoWhite
- OscarsSoWhite -BLM
