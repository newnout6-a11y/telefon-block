package com.antispam.blocker.domain.scoring

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager

class InstalledAppScanner(private val context: Context) {

    data class ScanResult(
        val hasBankingApp: Boolean = false,
        val hasGovApp: Boolean = false,
        val hasMarketplaceApp: Boolean = false,
        val hasAdsApp: Boolean = false,
        val hasDeliveryApp: Boolean = false,
        val hasMessengerApp: Boolean = false,
        val hasVpnAntivirusApp: Boolean = false,
        val hasKidsApp: Boolean = false,
        val hasCryptoApp: Boolean = false,
        val detectedCategories: List<String> = emptyList()
    )

    fun scan(): ScanResult {
        val pm = context.packageManager
        val installedApps = pm.getInstalledApplications(0)
            .map { it.packageName }
            .toSet()

        val categories = mutableListOf<String>()

        val hasBanking = BANK_PACKAGES.any { it in installedApps }
        if (hasBanking) categories.add("banking")

        val hasGov = GOV_PACKAGES.any { it in installedApps }
        if (hasGov) categories.add("gov")

        val hasMarketplace = MARKETPLACE_PACKAGES.any { it in installedApps }
        if (hasMarketplace) categories.add("marketplace")

        val hasAds = ADS_PACKAGES.any { it in installedApps }
        if (hasAds) categories.add("ads")

        val hasDelivery = DELIVERY_PACKAGES.any { it in installedApps }
        if (hasDelivery) categories.add("delivery")

        val hasMessenger = MESSENGER_PACKAGES.any { it in installedApps }
        if (hasMessenger) categories.add("messenger")

        val hasVpnAntivirus = VPN_ANTIVIRUS_PACKAGES.any { it in installedApps }
        if (hasVpnAntivirus) categories.add("vpn_antivirus")

        val hasKids = KIDS_PACKAGES.any { it in installedApps }
        if (hasKids) categories.add("kids")

        val hasCrypto = CRYPTO_PACKAGES.any { it in installedApps }
        if (hasCrypto) categories.add("crypto")

        return ScanResult(
            hasBankingApp = hasBanking,
            hasGovApp = hasGov,
            hasMarketplaceApp = hasMarketplace,
            hasAdsApp = hasAds,
            hasDeliveryApp = hasDelivery,
            hasMessengerApp = hasMessenger,
            hasVpnAntivirusApp = hasVpnAntivirus,
            hasKidsApp = hasKids,
            hasCryptoApp = hasCrypto,
            detectedCategories = categories
        )
    }

    fun enrichProfile(base: UserProfileVector): UserProfileVector {
        val scan = scan()
        var digital = base.digitalActivity
        var ads = base.adsActivity
        var awareness = base.awarenessLevel

        if (scan.hasBankingApp) digital += 10
        if (scan.hasGovApp) digital += 5
        if (scan.hasMarketplaceApp) { digital += 5; ads += 5 }
        if (scan.hasAdsApp) ads += 10
        if (scan.hasDeliveryApp) ads += 5
        if (scan.hasVpnAntivirusApp) awareness += 10
        if (scan.hasCryptoApp) { digital += 10; awareness += 5 }

        return base.copy(
            digitalActivity = digital.coerceIn(0f, 100f),
            adsActivity = ads.coerceIn(0f, 100f),
            awarenessLevel = awareness.coerceIn(0f, 100f)
        )
    }

    companion object {
        private val BANK_PACKAGES = setOf(
            "ru.sberbankmobile", "com.idamob.tinkoff.android", "ru.vtb.mobile",
            "ru.alfabank.mobile.android", "ru.gazprombank.mobile", "ru.rshb.v1",
            "ru.mkb.mobile", "ru.psbc.mbank", "ru.rosbank.mobile",
            "ru.otp.mobile", "ru.raiffeisen", "ru.bss.mobile",
            "com.samsung.android.spay", "com.google.android.apps.walletnfcrel"
        )

        private val GOV_PACKAGES = setOf(
            "ru.rostelecom.gosuslugi", "ru.gosuslugi", "ru.minsvyaz.gosuslugi"
        )

        private val MARKETPLACE_PACKAGES = setOf(
            "com.wildberries.ru", "com.ozon.android", "ru.beru.android",
            "com.sbermarket", "ru.megamarket"
        )

        private val ADS_PACKAGES = setOf(
            "com.avito.android", "ru.yula"
        )

        private val DELIVERY_PACKAGES = setOf(
            "ru.yandex.eda", "com.deliveryclub", "com.samokat.app",
            "ru.sbermarket"
        )

        private val MESSENGER_PACKAGES = setOf(
            "com.whatsapp", "org.telegram.messenger", "com.viber.voip",
            "vk.messenger.android", "com.tencent.mm"
        )

        private val VPN_ANTIVIRUS_PACKAGES = setOf(
            "com.kaspersky.mobile", "com.drweb.pro", "com.drweb",
            "com.wireguard.android", "de.blinkt.openvpn",
            "net.mztech.vpn", "com.vpn.proxy.master"
        )

        private val KIDS_PACKAGES = setOf(
            "ru.rtr.malysh", "com.foxandsheep.kidsapps",
            "com.budgestudios.googleplay", "com.duolingo"
        )

        private val CRYPTO_PACKAGES = setOf(
            "com.binance.dev", "com.bybit.app", "com.coinbase.android",
            "com.wallet.crypto.trustapp"
        )
    }
}
