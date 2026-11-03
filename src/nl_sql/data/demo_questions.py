"""Fixed evaluation questions for the synthetic demo database, each with gold SQL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoQuestion:
    """One plain-English question and the SQL whose result defines the right answer.

    The skill tag groups questions in reports so a reader can see which kinds of
    question the agent handles, not only an overall score.
    """

    question_id: str
    question: str
    gold_sql: str
    skill: str


# Several questions probe the domain hints on purpose: "shipped" totals, JSON
# metadata, and historical versus current prices are where naive SQL goes wrong.
DEMO_QUESTIONS: tuple[DemoQuestion, ...] = (
    DemoQuestion("q01", "How many orders are there?", "SELECT COUNT(*) FROM orders", "count"),
    DemoQuestion(
        "q02",
        "How many orders have shipped?",
        "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
        "filter",
    ),
    DemoQuestion(
        "q03",
        "What is the total value of shipped orders?",
        "SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi "
        "JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'shipped'",
        "business rule",
    ),
    DemoQuestion(
        "q04",
        "How many orders are expedited?",
        "SELECT COUNT(*) FROM orders WHERE json_extract(metadata, '$.expedited') = 1",
        "JSON field",
    ),
    DemoQuestion(
        "q05",
        "How many expedited orders have shipped?",
        "SELECT COUNT(*) FROM orders WHERE status = 'shipped' "
        "AND json_extract(metadata, '$.expedited') = 1",
        "JSON field",
    ),
    DemoQuestion(
        "q06",
        "How many orders came through the partner channel?",
        "SELECT COUNT(*) FROM orders WHERE json_extract(metadata, '$.channel') = 'partner'",
        "JSON field",
    ),
    DemoQuestion(
        "q07",
        "Which customer has spent the most on shipped orders?",
        "SELECT c.customer_name FROM customers c "
        "JOIN orders o ON o.customer_id = c.customer_id "
        "JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status = 'shipped' "
        "GROUP BY c.customer_id ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1",
        "join + ranking",
    ),
    DemoQuestion(
        "q08",
        "Which region has the highest shipped order value, and what is that value?",
        "SELECT c.region, SUM(oi.quantity * oi.unit_price) AS value FROM customers c "
        "JOIN orders o ON o.customer_id = c.customer_id "
        "JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status = 'shipped' "
        "GROUP BY c.region ORDER BY value DESC LIMIT 1",
        "join + ranking",
    ),
    DemoQuestion(
        "q09",
        "What is the total quantity sold for each product across all orders?",
        "SELECT p.product_name, SUM(oi.quantity) FROM products p "
        "JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.product_id",
        "group by",
    ),
    DemoQuestion(
        "q10",
        "What is the average order value across all orders?",
        "SELECT AVG(order_value) FROM (SELECT SUM(quantity * unit_price) AS order_value "
        "FROM order_items GROUP BY order_id)",
        "nested aggregate",
    ),
    DemoQuestion(
        "q11",
        "What was the total value of order O101?",
        "SELECT SUM(quantity * unit_price) FROM order_items WHERE order_id = 'O101'",
        "lookup",
    ),
    DemoQuestion(
        "q12",
        "List the pending orders with the name of the customer who placed each one.",
        "SELECT o.order_id, c.customer_name FROM orders o "
        "JOIN customers c ON c.customer_id = o.customer_id WHERE o.status = 'pending'",
        "join",
    ),
    DemoQuestion(
        "q13",
        "Which customer placed the most recent order?",
        "SELECT c.customer_name FROM orders o JOIN customers c "
        "ON c.customer_id = o.customer_id ORDER BY o.order_date DESC LIMIT 1",
        "ordering",
    ),
    DemoQuestion(
        "q14",
        "How many customers have no pending orders?",
        "SELECT COUNT(*) FROM customers WHERE customer_id NOT IN "
        "(SELECT customer_id FROM orders WHERE status = 'pending')",
        "anti-join",
    ),
    DemoQuestion(
        "q15",
        "What would the shipped orders be worth at today's product prices?",
        "SELECT SUM(oi.quantity * p.current_price) FROM order_items oi "
        "JOIN orders o ON o.order_id = oi.order_id "
        "JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'shipped'",
        "business rule",
    ),
)
