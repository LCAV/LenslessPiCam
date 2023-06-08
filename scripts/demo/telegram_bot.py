"""

Telegram bot to interface with lensless camera setup.

Create a "secrets.py" file inside the ``lensless`` package
and put your Telegram bot token in it.

TODO: pass token as command line argument? and rpi config?

"""

import logging
import numpy as np
import os
from PIL import Image
import shutil
from lensless.secrets import TELEGRAM_BOT

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


TOKEN = TELEGRAM_BOT
INPUT_FP = "user_photo.jpg"
OUTPUT_FOLDER = "demo_lensless_recon"


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Send a picture to send it the lensless camera setup for reconstruction."
    )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Acknowledges photo receipt."""

    # user = update.message.from_user

    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(INPUT_FP)

    # get shape of picture
    img = np.array(Image.open(INPUT_FP))
    await update.message.reply_text("Got photo of resolution: " + str(img.shape))
    await update.message.reply_text("Processing...")

    # call python script
    os.system(f"python scripts/demo.py plot=False fp={INPUT_FP} output={OUTPUT_FOLDER}")

    # return reconstructed file
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")

    img = np.array(Image.open(OUTPUT_FP))
    await update.message.reply_text("Output resolution: " + str(img.shape))

    await update.message.reply_photo(OUTPUT_FP)
    # await update.message.reply_photo(INPUT_FP)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

    # delete photo
    if os.path.exists(INPUT_FP):
        os.remove(INPUT_FP)

    # delete non-empty folder
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
