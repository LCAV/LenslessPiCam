"""

Telegram bot to interface with lensless camera setup.

"""

import hydra
import logging
import numpy as np
import os
from PIL import Image
import shutil

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


TOKEN = None
RPI_USERNAME = None
RPI_HOSTNAME = None
RPI_LENSED_USERNAME = None
RPI_LENSED_HOSTNAME = None
INPUT_FP = "user_photo.jpg"
OUTPUT_FOLDER = "demo_lensless_recon"
BUSY = False
supported_algos = ["fista", "admm", "unrolled"]
supported_input = ["mnist", "thumb"]
BRIGHTNESS = 100


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def check_algo(algo):

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
        rf"Hi {user.mention_html()}! Try one of the commands to reconstruct one of the available content, or send a (portrait) picture.",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Send a picture to the lensless camera setup for reconstruction."
    )


async def algo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""

    # get text from user
    try:
        algo = update.message.text.split(" ")[1]
    except IndexError:
        await update.message.reply_text("Please specify an algorithm from: ", supported_algos)
        return

    if check_algo(algo):

        # reconstruct
        await update.message.reply_text(f"Reconstructing with {algo}...")
        os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
        OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
        await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
        # img = np.array(Image.open(OUTPUT_FP))
        # await update.message.reply_text("Output resolution: " + str(img.shape))

    else:

        await update.message.reply_text("Unsupported algorithm : " + algo)


async def fista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    algo = "fista"
    os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
    await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
    # img = np.array(Image.open(OUTPUT_FP))
    # await update.message.reply_text("Output resolution: " + str(img.shape))


async def admm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    algo = "admm"
    os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
    await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
    # img = np.array(Image.open(OUTPUT_FP))
    # await update.message.reply_text("Output resolution: " + str(img.shape))


async def unrolled(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    algo = "unrolled"
    os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
    await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
    # img = np.array(Image.open(OUTPUT_FP))
    # await update.message.reply_text("Output resolution: " + str(img.shape))


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

    # get shape of picture, must be portrait
    await photo_file.download_to_drive(INPUT_FP)
    img = np.array(Image.open(INPUT_FP))

    # -- check if portrait
    if img.shape[0] < img.shape[1]:
        await update.message.reply_text("Please send a portrait photo.")
        return
    else:
        await update.message.reply_text("Got photo of resolution: " + str(img.shape))

    if check_algo(algo):

        if BUSY:
            await update.message.reply_text("Busy processing previous request.")
            return
        BUSY = True

        # # call python script for full process
        # os.system(f"python scripts/demo.py plot=False fp={INPUT_FP} output={OUTPUT_FOLDER}")

        # -- send to display
        os.system(
            f"python scripts/remote_display.py fp={INPUT_FP} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
        )
        await update.message.reply_text("Image sent to display.")

        await take_picture_and_reconstruct(update, context, algo)

    else:

        await update.message.reply_text(
            f"Unsupported algorithm: {algo}. Please specify from: {supported_algos}"
        )

    BUSY = False


async def take_picture_and_reconstruct(
    update: Update, context: ContextTypes.DEFAULT_TYPE, algo
) -> None:

    # -- measurement
    os.system(
        f"python scripts/remote_capture.py plot=False rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text("Took picture.")

    # -- reconstruct
    await update.message.reply_text(f"Reconstructing with {algo}...")
    os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
    await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})")
    # img = np.array(Image.open(OUTPUT_FP))
    # await update.message.reply_text("Output resolution: " + str(img.shape))

    # -- send picture of raw measurement
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "raw_data_plot.png")
    await update.message.reply_photo(OUTPUT_FP, caption="Raw measurement")

    # -- send picture of setup (lensed)
    os.system(
        f"python scripts/remote_capture.py rpi.username={RPI_LENSED_USERNAME} rpi.hostname={RPI_LENSED_HOSTNAME} plot=False capture.bayer=False capture.down=8 capture.raw_data_fn=lensed capture.awb_gains=null"
    )
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "lensed.png")
    await update.message.reply_photo(OUTPUT_FP, caption="Picture of setup")


async def mnist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Use one of the input images
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    if BUSY:
        await update.message.reply_text("Busy processing previous request.")
        return
    BUSY = True
    algo = "admm"
    vshift = -10
    brightness = 100

    # -- send to display
    os.system(
        f"python scripts/remote_display.py fp=data/original/mnist_3.png display.vshift={vshift} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text("Image sent to display.")

    await take_picture_and_reconstruct(update, context, algo)
    BUSY = False


async def thumb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Use one of the input images
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    if BUSY:
        await update.message.reply_text("Busy processing previous request.")
        return
    BUSY = True
    algo = "admm"
    vshift = -10
    brightness = 80

    # -- send to display
    os.system(
        f"python scripts/remote_display.py fp=data/original/thumbs_up.png display.vshift={vshift} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text("Image sent to display.")

    await take_picture_and_reconstruct(update, context, algo)
    BUSY = False


async def face_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Use one of the input images
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    if BUSY:
        await update.message.reply_text("Busy processing previous request.")
        return
    BUSY = True
    algo = "admm"
    vshift = -10
    brightness = 100

    # -- send to display
    os.system(
        f"python scripts/remote_display.py fp=data/original/face.jpg display.vshift={vshift} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text("Image sent to display.")

    await take_picture_and_reconstruct(update, context, algo)
    BUSY = False


async def psf_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    Measure PSF through screen
    """

    global BUSY

    if BUSY:
        await update.message.reply_text("Busy processing previous request.")
        return
    BUSY = True
    vshift = -15
    psf_size = 10

    # -- send to display
    os.system(
        f"python scripts/remote_display.py display.psf={psf_size} display.vshift={vshift} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text(f"PSF of {psf_size}x{psf_size} pixels set on display.")

    # -- measurement
    os.system(
        f"python scripts/remote_capture.py -cn demo_measure_psf rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "raw_data.png")
    await update.message.reply_photo(OUTPUT_FP, caption="PSF (zoom in to see caustic pattern)")

    BUSY = False


async def brightness_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    Set brightness, re-capture, and reconstruct.
    """

    global BUSY

    # get text from user
    brightness = None
    try:
        brightness = float(update.message.text.split(" ")[1])
        assert brightness > 0 and brightness <= 100
    except IndexError or AssertionError:
        await update.message.reply_text(
            "Please specify brightness within (0, 100],\ne.g /brightness 50"
        )
        return
    except AssertionError:
        await update.message.reply_text(
            "Please specify brightness within (0, 100],\ne.g /brightness 50"
        )
        return

    if BUSY:
        await update.message.reply_text("Busy processing previous request.")
        return
    BUSY = True

    # -- resend to display
    os.system(
        f"python scripts/remote_display.py fp={INPUT_FP} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text("Image sent to display.")

    algo = "admm"
    await take_picture_and_reconstruct(update, context, algo)

    BUSY = False


@hydra.main(version_base=None, config_path="../../configs", config_name="telegram_demo")
def main(config) -> None:
    """Start the bot."""

    global TOKEN, RPI_USERNAME, RPI_HOSTNAME, RPI_LENSED_USERNAME, RPI_LENSED_HOSTNAME

    TOKEN = config.token
    RPI_USERNAME = config.rpi_username
    RPI_HOSTNAME = config.rpi_hostname
    RPI_LENSED_USERNAME = config.rpi_lensed_username
    RPI_LENSED_HOSTNAME = config.rpi_lensed_hostname

    assert TOKEN is not None
    assert RPI_USERNAME is not None
    assert RPI_HOSTNAME is not None
    assert RPI_LENSED_USERNAME is not None
    assert RPI_LENSED_HOSTNAME is not None

    # make output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mnist", mnist_command))
    application.add_handler(CommandHandler("thumb", thumb_command))
    application.add_handler(CommandHandler("face", face_command))
    application.add_handler(CommandHandler("psf", psf_command))
    application.add_handler(CommandHandler("brightness", brightness_command))

    # application.add_handler(CommandHandler("algo", algo_command))
    application.add_handler(CommandHandler("fista", fista))
    application.add_handler(CommandHandler("admm", admm))
    application.add_handler(CommandHandler("unrolled", unrolled))

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
