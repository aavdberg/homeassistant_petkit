"""Native BLE coordinator for PetKit water fountains.

Provides direct Bluetooth communication with PetKit water fountains
(W4/W5/CTW2/CTW3) using Home Assistant's bluetooth component, which
transparently supports both local Bluetooth adapters and ESPHome
Bluetooth proxies.

This module is only used when the user enables 'Local BLE' mode in the
integration options. It runs independently of the cloud coordinator and
updates the WaterFountain entities directly from BLE data.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pypetkitapi import (
    BLE_FOUNTAIN_NAME_PREFIXES,
    BLE_NOTIFY_UUID,
    BLE_WRITE_UUID,
    LocalFountainBleProtocol,
)

from homeassistant.components import bluetooth
from homeassistant.components.bluetooth import BluetoothServiceInfoBleak
from homeassistant.core import HomeAssistant

if TYPE_CHECKING:
    from bleak import BleakClient

_LOGGER = logging.getLogger(__name__)

# Timeout (seconds) waiting for a BLE notification after sending a command
BLE_NOTIFY_TIMEOUT = 10.0
# Delay (seconds) between commands in the init sequence
BLE_CMD_DELAY = 0.5


def is_petkit_fountain(service_info: BluetoothServiceInfoBleak) -> bool:
    """Return True if a BLE device is a supported PetKit fountain."""
    return service_info.name is not None and service_info.name.startswith(
        BLE_FOUNTAIN_NAME_PREFIXES
    )


def discovered_fountains(hass: HomeAssistant) -> list[BluetoothServiceInfoBleak]:
    """Return all currently discovered PetKit fountain BLE devices."""
    return [
        info
        for info in bluetooth.async_discovered_service_info(hass)
        if is_petkit_fountain(info)
    ]


def _alias_from_name(device_name: str) -> str:
    """Derive the device alias (e.g. 'CTW3', 'W5') from the BLE device name."""
    for prefix in (
        "Petkit_CTW3",
        "Petkit_CTW2",
        "Petkit_W5C",
        "Petkit_W5N",
        "Petkit_W5",
        "Petkit_W4XUVC",
        "Petkit_W4X",
    ):
        if device_name.startswith(prefix):
            return prefix.replace("Petkit_", "")
    return ""


class FountainBleClient:
    """Manages a direct BLE connection to a single PetKit fountain.

    Handles the full lifecycle:
    1. Obtain BLEDevice from HA bluetooth scanner
    2. Establish connection (with retry via bleak_retry_connector)
    3. Run initialization handshake
    4. Poll full status (CMD 230)
    5. Send control commands (mode, config)
    6. Disconnect cleanly
    """

    def __init__(
        self,
        hass: HomeAssistant,
        mac_address: str,
        device_name: str,
    ) -> None:
        """Initialize the BLE client for a specific fountain."""
        self.hass = hass
        self.mac_address = mac_address
        self.device_name = device_name
        self._alias = _alias_from_name(device_name)
        self._protocol = LocalFountainBleProtocol(self._alias)
        self._client: BleakClient | None = None
        self._notification_event: asyncio.Event = asyncio.Event()
        self._last_status: dict[str, Any] | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def async_get_status(self) -> dict[str, Any] | None:
        """Connect to the fountain, read full status, disconnect.

        Returns the parsed status dict, or None on failure.
        """
        try:
            await self._connect()
            await self._run_init_sequence()
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("BLE status read failed for %s: %s", self.mac_address, err)
            await self._disconnect()
            return None
        else:
            status = await self._read_full_status()
            await self._disconnect()
            return status

    async def async_set_mode(self, power_state: int, mode: int) -> bool:
        """Connect, send a mode change command, disconnect."""
        try:
            await self._connect()
            await self._run_init_sequence()
            cmd = self._protocol.build_set_mode_command(power_state, mode)
            await self._write(cmd)
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("BLE set_mode failed for %s: %s", self.mac_address, err)
            await self._disconnect()
            return False
        else:
            await self._disconnect()
            return True

    async def async_set_config(self, config_data: list[int]) -> bool:
        """Connect, send a config update command, disconnect."""
        try:
            await self._connect()
            await self._run_init_sequence()
            cmd = self._protocol.build_set_config_command(config_data)
            await self._write(cmd)
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("BLE set_config failed for %s: %s", self.mac_address, err)
            await self._disconnect()
            return False
        else:
            await self._disconnect()
            return True

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def _connect(self) -> None:
        """Establish a BLE connection using HA bluetooth + bleak_retry_connector."""
        from bleak import BleakClient
        from bleak_retry_connector import establish_connection

        ble_device = bluetooth.async_ble_device_from_address(
            self.hass, self.mac_address, connectable=True
        )
        if ble_device is None:
            raise RuntimeError(
                f"BLE device {self.mac_address} not found in HA scanner. "
                "Ensure Bluetooth is enabled and the fountain is powered on."
            )

        self._protocol = LocalFountainBleProtocol(self._alias)
        self._last_status = None

        self._client = await establish_connection(
            BleakClient,
            ble_device,
            self.device_name,
            disconnected_callback=self._on_disconnected,
        )
        await self._client.start_notify(BLE_NOTIFY_UUID, self._on_notification)
        _LOGGER.debug("BLE connected to %s (%s)", self.device_name, self.mac_address)

    async def _disconnect(self) -> None:
        """Cleanly disconnect from the fountain."""
        if self._client and self._client.is_connected:
            try:
                await self._client.stop_notify(BLE_NOTIFY_UUID)
                await self._client.disconnect()
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("BLE disconnect error for %s: %s", self.mac_address, err)
        self._client = None

    def _on_disconnected(self, _client: Any) -> None:
        _LOGGER.debug("BLE disconnected from %s", self.mac_address)

    # -----------------------------------------------------------------------
    # Protocol sequence
    # -----------------------------------------------------------------------

    async def _run_init_sequence(self) -> None:
        """Run the BLE initialization handshake."""
        # Step 1: Send CMD 213 (request device ID)
        for cmd in self._protocol.get_init_commands():
            await self._write(cmd)

        # Wait for device ID notification
        await self._wait_for_notification()

        if not self._protocol.device_id_received:
            _LOGGER.warning(
                "Device ID not received for %s; proceeding anyway", self.mac_address
            )

        # Step 2: Send CMD 73 (auth), CMD 86 (sync), CMD 84 (set time)
        for cmd in self._protocol.complete_init_commands():
            await self._write(cmd)
            await asyncio.sleep(BLE_CMD_DELAY)

    async def _read_full_status(self) -> dict[str, Any] | None:
        """Send CMD 230 and wait for the status notification."""
        self._notification_event.clear()
        self._last_status = None
        await self._write(self._protocol.get_status_command())
        await self._wait_for_notification()
        return self._last_status

    # -----------------------------------------------------------------------
    # BLE I/O helpers
    # -----------------------------------------------------------------------

    async def _write(self, data: bytearray) -> None:
        if self._client is None or not self._client.is_connected:
            raise RuntimeError("BLE client is not connected")
        await self._client.write_gatt_char(BLE_WRITE_UUID, data, response=False)

    async def _wait_for_notification(self) -> None:
        """Wait up to BLE_NOTIFY_TIMEOUT seconds for a BLE notification."""
        self._notification_event.clear()
        try:
            await asyncio.wait_for(self._notification_event.wait(), BLE_NOTIFY_TIMEOUT)
        except asyncio.TimeoutError:
            _LOGGER.debug("BLE notification timeout for %s", self.mac_address)

    def _on_notification(self, _sender: Any, data: bytearray) -> None:
        """Handle incoming BLE notification."""
        result = self._protocol.handle_notification(data)
        if result is not None:
            self._last_status = result
        # Signal that at least one notification arrived
        self._notification_event.set()
