# Verifying the security state of an ESP32-S3 metering node

These steps read back the one-time-programmable eFuses, which are the
authoritative record of what the chip enforces. They do not modify the device.

```bash
pip install esptool            # provides espefuse.py
espefuse.py --chip esp32s3 --port <PORT> summary > efuse_summary.txt
```

In `efuse_summary.txt` check:

| What | eFuse field | Expected |
|---|---|---|
| Secure Boot V2 enabled | `SECURE_BOOT_EN` | `True` |
| Secure Boot digest present | `KEY_PURPOSE_n` | one block = `SECURE_BOOT_DIGEST0` |
| Flash encryption enabled | `SPI_BOOT_CRYPT_CNT` | odd number of bits set (e.g. `0b001`) |
| Flash-encryption key | `KEY_PURPOSE_n` | one block = `XTS_AES_128_KEY` |
| Record-authentication key | `KEY_PURPOSE_n` | one block = `HMAC_UP` |
| Keys not readable by software | `RD_DIS` | bits set for the flash-encryption and HMAC key blocks (shown as `= ?? ?? ...` read-protected) |

Also confirm the application image is signed:

```bash
espsecure.py verify_signature --version 2 --keyfile <public_key.pem> build/<app>.bin
```

Keep `efuse_summary.txt` (it contains no secret values) as evidence of the
device configuration; it may be added to this folder.

## Provisioning the record-authentication key

```bash
python -c "import os; open('hmac_key.bin','wb').write(os.urandom(32))"   # offline workstation
espefuse.py --chip esp32s3 --port <PORT> burn_key BLOCK_KEY4 hmac_key.bin HMAC_UP
# register hmac_key.bin in the server key store, then securely delete the local copy
```
