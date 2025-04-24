# Intro
This file has instructions for how to opporate weekly report_generator.ipynb and its associated notebooks.

# Concept
The weekly report is designed to be a way to look at the health and function of the EPMT database over a period of time. 
The weekly report is dated based upon the date that ends a seven day period. It produces a master histogram of 27 different EPMT metrics, a chart showing flux (bytes per unit time) for each node, a pie chart displaying the top 5 heaviest users for that week, and a table of total and average values for key metrics. Future functionality will allow this to table to also display how those numbers relate to other weekly reports.

# Functionality
The script report_generator.ipynb is designed to be easily run. Open in jupyter and run all cells, and the script will create a folder wherever you are that is labeled weekly_report_<date> and populate it with all 4 plots. If you rerun on the same day, those plots will be replaced with the new ones. In order to choose a specific date for the weekly report, instead of just the current date, uncommonet the third cell and run it instead of the second, after changing `start_date = datetime.datetime(year, month, day, hour)` to your prefered date. It does not matter if cell 2 is accidentally run first.