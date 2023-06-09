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
BUSY = False


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def check_algo(algo):

    supported_algos = ["fista", "admm"]
    if algo not in supported_algos:
        return False
    else:
        return True


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


async def algo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""

    # get text from user
    try:
        algo = update.message.text.split(" ")[1]
    except ValueError:
        await update.message.reply_text("Please specify an algorithm.")
        return

    if check_algo(algo):

        # reconstruct
        await update.message.reply_text(f"Reconstructing with {algo}...")
        os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
        OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
        await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
        img = np.array(Image.open(OUTPUT_FP))
        await update.message.reply_text("Output resolution: " + str(img.shape))

    else:

        await update.message.reply_text("Unsupported algorithm : " + algo)


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Get photo from user
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    photo_file = await update.message.photo[-1].get_file()

    algo = update.message.caption
    if algo is not None:
        algo = algo.lower()
    else:
        algo = "admm"

    if check_algo(algo):

        if BUSY:
            await update.message.reply_text("Busy processing previous request.")
            return
        BUSY = True

        await photo_file.download_to_drive(INPUT_FP)

        # get shape of picture
        img = np.array(Image.open(INPUT_FP))
        await update.message.reply_text("Got photo of resolution: " + str(img.shape))

        # # call python script for full process
        # os.system(f"python scripts/demo.py plot=False fp={INPUT_FP} output={OUTPUT_FOLDER}")

        # -- send to display
        os.system(f"python scripts/remote_display.py fp={INPUT_FP}")
        await update.message.reply_text("Image sent to display.")

        # copy picture of setups
        # TODO remote capture of low resolution RGB image

        # -- measurement
        os.system("python scripts/remote_capture.py plot=False")
        OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "raw.png")
        await update.message.reply_photo(OUTPUT_FP, caption="Raw measurement")

        # -- reconstruct
        await update.message.reply_text(f"Reconstructing with {algo}...")
        os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
        OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
        await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
        img = np.array(Image.open(OUTPUT_FP))
        await update.message.reply_text("Output resolution: " + str(img.shape))

    else:

        await update.message.reply_text("Unsupported algorithm : " + algo)

    BUSY = False


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("algo", algo_command))

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
