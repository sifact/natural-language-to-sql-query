# demo_questions.py

Fixed plain-English questions about the synthetic demo shop, each paired with gold SQL.

* Defines `DemoQuestion`: an ID, the question, the gold SQL, and a skill tag for reports.
* Lists 15 questions in `DEMO_QUESTIONS`, from simple counts to joins, rankings and anti-joins.
* Deliberately probes the domain hints: shipped-only totals, JSON metadata, and historical versus current prices.
* Gold SQL defines the right answer by its executed result, so any equivalent query scores as correct.
* Holds only synthetic data, so questions and SQL can be published in reports and on the About page.
