from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

import llm_to_analyse
import portfolio_monitor

# ---------------------------------------------------------------------
# Two-way telegram bot for the stock agent.
# Chat messages go to the same graph the streamlit app uses (llm_to_analyse).
# Run with:  python telegram_bot.py   - it replies only while it is running.
# ---------------------------------------------------------------------

# windows consoles default to cp1252, the agents' prints of ₹ / emoji would crash a handler
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # only this chat may use the bot

TELEGRAM_MAX_LEN = 4000                      # telegram limit is 4096 characters per message

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)   # polling request urls contain the token
log = logging.getLogger("telegram_bot")

# the graphs are synchronous and slow, so they run in a worker thread;
# the locks stop two messages from using the same conversation / monitor at once
agent_lock = asyncio.Lock()
monitor_lock = asyncio.Lock()


async def send_long(bot, chat_id, text):
    chunks = [text[i:i + TELEGRAM_MAX_LEN] for i in range(0, len(text), TELEGRAM_MAX_LEN)] or ["(empty reply)"]
    for chunk in chunks:
        # LLM replies use **bold**, telegram's Markdown uses *bold*; broken markdown falls back to plain text
        try:
            await bot.send_message(chat_id=chat_id, text=chunk.replace("**", "*"), parse_mode="Markdown")
        except BadRequest:
            await bot.send_message(chat_id=chat_id, text=chunk)


def ask_agent(text, chat_id, pending):
    # same flow as interface.py: resume an interrupt, or start a new turn
    config = {"configurable": {"thread_id": f"telegram_{chat_id}"}}
    if pending:
        result = llm_to_analyse.graph.invoke(Command(resume=text), config)
    else:
        result = llm_to_analyse.graph.invoke({"messages": [HumanMessage(text)], "decision": ""}, config)
    if "__interrupt__" in result:
        return result["__interrupt__"][-1].value, True     # the agent is asking for more details
    return result["messages"][-1].content, False


def portfolio_snapshot():
    # live prices + P&L only, no LLM calls
    lines = []
    total_invested = total_value = 0
    for holding in portfolio_monitor.get_holdings_from_db():
        price = portfolio_monitor.get_current_price(holding["ticker"])
        if price is None:
            lines.append(f"{holding['ticker']}: no price available")
            continue
        pnl = portfolio_monitor.compute_pnl(holding, price)
        total_invested += pnl["invested"]
        total_value += pnl["current_value"]
        lines.append(f"{holding['stock_name']} ({holding['ticker']})\n"
                     f"{holding['stock_count']} x ₹{holding['bought_price']:,.2f} ➜ now ₹{price:,.2f}\n"
                     f"P&L ₹{pnl['pnl_amount']:,.2f} ({pnl['pnl_pct']:+.2f}%)")
    if not lines:
        return "Your portfolio is empty."
    total_pnl = total_value - total_invested
    total_pct = total_pnl / total_invested * 100 if total_invested else 0
    lines.append(f"Total: invested ₹{total_invested:,.2f}, now ₹{total_value:,.2f}\n"
                 f"P&L ₹{total_pnl:,.2f} ({total_pct:+.2f}%)")
    return "\n\n".join(lines)


def run_portfolio_check():
    output = portfolio_monitor.graph.invoke({"holdings": [], "results": []})
    return portfolio_monitor.format_telegram_message(output["results"])


# ---------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------

HELP_TEXT = (
    "I am your stock agent. You can:\n"
    "/portfolio - your holdings with live price and P&L\n"
    "/runnow - run the HOLD / SELL / BUY_MORE check on every holding\n"
    "/help - this message\n\n"
    "Or just write, e.g. \"should I sell TCS?\" or \"I bought 10 Infosys at 1500\"."
)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    snapshot = await asyncio.to_thread(portfolio_snapshot)
    await send_long(context.bot, update.effective_chat.id, snapshot)


async def runnow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if monitor_lock.locked():
        await update.message.reply_text("A portfolio check is already running.")
        return
    async with monitor_lock:
        await update.message.reply_text("🔄 Running the portfolio check, this takes about a minute per stock...")
        report = await asyncio.to_thread(run_portfolio_check)
    await send_long(context.bot, update.effective_chat.id, report)


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    async with agent_lock:
        pending = context.chat_data.get("pending", False)
        reply, pending = await asyncio.to_thread(ask_agent, update.message.text, chat_id, pending)
    context.chat_data["pending"] = pending
    await send_long(context.bot, chat_id, str(reply))


async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    log.error("Handler failed", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        context.chat_data["pending"] = False    # do not leave the chat stuck waiting for a resume
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=f"Sorry, something went wrong: {context.error}")


async def announce_start(app: Application):
    await app.bot.send_message(chat_id=CHAT_ID, text="🤖 Stock agent is online. Send /help to see what I can do.")


def main():
    if not TOKEN or not CHAT_ID:
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

    app = Application.builder().token(TOKEN).post_init(announce_start).build()

    # messages from any other chat match no handler and are ignored
    me = filters.Chat(chat_id=int(CHAT_ID))
    app.add_handler(CommandHandler(["start", "help"], help_command, filters=me))
    app.add_handler(CommandHandler("portfolio", portfolio, filters=me))
    app.add_handler(CommandHandler("runnow", runnow, filters=me))
    app.add_handler(MessageHandler(me & filters.TEXT & ~filters.COMMAND, text_message))
    app.add_error_handler(on_error)

    log.info("Bot is running, press Ctrl+C to stop")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
