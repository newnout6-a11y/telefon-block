package com.antispam.blocker.domain.scoring

data class UserProfileVector(
    val vulnerabilityScore: Float = 0.5f,
    val businessActivity: Float = 0.5f,
    val digitalActivity: Float = 0.5f,
    val adsActivity: Float = 0.5f,
    val spamTolerance: Float = 0.5f,
    val falseAlarmFear: Float = 0.5f,
    val awarenessLevel: Float = 0.5f,
    val hasForeignContacts: Boolean = false,
    val hasHomePhone: Boolean = false
) {
    fun toFeatureValues(): Pair<Float, Float> {
        val vulnerability = vulnerabilityScore / 100f
        val business = businessActivity / 100f
        return Pair(vulnerability.coerceIn(0f, 1f), business.coerceIn(0f, 1f))
    }

    fun warnThreshold(): Float = when {
        falseAlarmFear > 70 -> 0.45f
        spamTolerance > 70 -> 0.25f
        vulnerabilityScore > 70 -> 0.25f
        else -> DEFAULT_WARN_THRESHOLD
    }

    fun blockThreshold(): Float = when {
        falseAlarmFear > 70 -> 0.85f
        businessActivity > 70 -> 0.85f
        spamTolerance > 70 -> 0.55f
        vulnerabilityScore > 70 -> 0.55f
        else -> DEFAULT_BLOCK_THRESHOLD
    }

    companion object {
        const val DEFAULT_WARN_THRESHOLD = 0.35f
        const val DEFAULT_BLOCK_THRESHOLD = 0.7f

        fun fromQuestionnaire(answers: QuestionnaireAnswers): UserProfileVector {
            var vulnerability = 0f
            var business = 0f
            var digital = 0f
            var ads = 0f
            var spamTol = 0f
            var falseAlarm = 0f
            var awareness = 0f

            // Age
            when (answers.age) {
                AgeRange.UNDER_25 -> { awareness += 10 }
                AgeRange.RANGE_25_40 -> { awareness += 5 }
                AgeRange.RANGE_40_60 -> { vulnerability += 15 }
                AgeRange.OVER_60 -> { vulnerability += 40; awareness -= 10 }
            }

            // Occupation
            when (answers.occupation) {
                Occupation.OFFICE -> { business += 10 }
                Occupation.FREELANCE -> { business += 15 }
                Occupation.ENTREPRENEUR -> { business += 30; digital += 10 }
                Occupation.PENSIONER -> { vulnerability += 20 }
                Occupation.STUDENT -> { digital += 10; awareness += 5 }
                Occupation.OTHER -> {}
            }

            // Work calls
            when (answers.workCalls) {
                CallFrequency.YES -> business += 25
                CallFrequency.SOMETIMES -> business += 10
                CallFrequency.NO -> {}
            }

            // Mobile banking
            if (answers.mobileBanking) digital += 15

            // Bank apps count
            when (answers.bankAppCount) {
                BankAppCount.ZERO -> {}
                BankAppCount.ONE_TWO -> digital += 10
                BankAppCount.THREE_PLUS -> { digital += 20; vulnerability += 5 }
            }

            // Online purchases
            when (answers.onlinePurchases) {
                PurchaseFrequency.OFTEN -> digital += 15
                PurchaseFrequency.SOMETIMES -> digital += 5
                PurchaseFrequency.RARELY -> {}
                PurchaseFrequency.NO -> {}
            }

            // Marketplaces
            if (answers.usesMarketplaces) { digital += 10; ads += 5 }

            // Avito/ads
            when (answers.adsPosting) {
                PostFrequency.OFTEN -> ads += 25
                PostFrequency.SOMETIMES -> ads += 10
                PostFrequency.NO -> {}
            }

            // Delivery
            when (answers.deliveryUsage) {
                PostFrequency.OFTEN -> ads += 15
                PostFrequency.SOMETIMES -> ads += 5
                PostFrequency.NO -> {}
            }

            // Messengers
            if (answers.usesWhatsApp) digital += 5
            if (answers.usesTelegram) { digital += 5; awareness += 5 }
            if (answers.usesViber) {}

            // Messenger calls from strangers
            when (answers.messengerStrangerCalls) {
                CallFrequency.YES -> vulnerability += 5
                CallFrequency.SOMETIMES -> {}
                CallFrequency.NO -> {}
            }

            // Foreign contacts
            when (answers.foreignContacts) {
                ForeignContacts.OFTEN -> {}
                ForeignContacts.RARELY -> {}
                ForeignContacts.NO -> {}
            }

            // Home phone
            // (handled via hasHomePhone flag)

            // Spam frequency
            when (answers.spamFrequency) {
                SpamFrequency.EVERY_DAY -> spamTol += 10
                SpamFrequency.SEVERAL_WEEK -> spamTol += 5
                SpamFrequency.RARELY -> {}
                SpamFrequency.ALMOST_NONE -> {}
            }

            // Scam experience
            when (answers.scamExperience) {
                ScamExperience.LOST_MONEY -> { vulnerability += 30; falseAlarm -= 10 }
                ScamExperience.CAUGHT_IN_TIME -> { vulnerability += 10; awareness += 10 }
                ScamExperience.NO -> {}
            }

            // Scam awareness
            when (answers.scamAwareness) {
                AwarenessLevel.GOOD -> awareness += 25
                AwarenessLevel.SUPERFICIAL -> {}
                AwarenessLevel.NONE -> { vulnerability += 15; awareness -= 15 }
            }

            // Priority: miss call vs hear spam
            when (answers.protectionPriority) {
                ProtectionPriority.DONT_MISS -> falseAlarm += 25
                ProtectionPriority.BALANCE -> {}
                ProtectionPriority.NO_SPAM -> spamTol += 25
            }

            // Answer stranger calls
            when (answers.answerStrangers) {
                AnswerStrangers.ALWAYS -> business += 10
                AnswerStrangers.SOMETIMES -> {}
                AnswerStrangers.NEVER -> spamTol += 10
            }

            // Auto-block vs warn
            when (answers.autoBlockPreference) {
                AutoBlockPreference.BLOCK -> spamTol += 20
                AutoBlockPreference.WARN -> {}
                AutoBlockPreference.DECIDE_EACH -> falseAlarm += 10
            }

            // Previous false block
            if (answers.hadFalseBlock) falseAlarm += 30

            return UserProfileVector(
                vulnerabilityScore = vulnerability.coerceIn(0f, 100f),
                businessActivity = business.coerceIn(0f, 100f),
                digitalActivity = digital.coerceIn(0f, 100f),
                adsActivity = ads.coerceIn(0f, 100f),
                spamTolerance = spamTol.coerceIn(0f, 100f),
                falseAlarmFear = falseAlarm.coerceIn(0f, 100f),
                awarenessLevel = awareness.coerceIn(0f, 100f),
                hasForeignContacts = answers.foreignContacts != ForeignContacts.NO,
                hasHomePhone = answers.hasHomePhone
            )
        }
    }
}

data class QuestionnaireAnswers(
    val age: AgeRange = AgeRange.RANGE_25_40,
    val occupation: Occupation = Occupation.OFFICE,
    val workCalls: CallFrequency = CallFrequency.SOMETIMES,
    val mobileBanking: Boolean = true,
    val bankAppCount: BankAppCount = BankAppCount.ONE_TWO,
    val onlinePurchases: PurchaseFrequency = PurchaseFrequency.SOMETIMES,
    val usesMarketplaces: Boolean = true,
    val adsPosting: PostFrequency = PostFrequency.NO,
    val deliveryUsage: PostFrequency = PostFrequency.SOMETIMES,
    val usesWhatsApp: Boolean = true,
    val usesTelegram: Boolean = true,
    val usesViber: Boolean = false,
    val messengerStrangerCalls: CallFrequency = CallFrequency.NO,
    val foreignContacts: ForeignContacts = ForeignContacts.NO,
    val hasHomePhone: Boolean = false,
    val spamFrequency: SpamFrequency = SpamFrequency.RARELY,
    val scamExperience: ScamExperience = ScamExperience.NO,
    val scamAwareness: AwarenessLevel = AwarenessLevel.SUPERFICIAL,
    val protectionPriority: ProtectionPriority = ProtectionPriority.BALANCE,
    val answerStrangers: AnswerStrangers = AnswerStrangers.SOMETIMES,
    val autoBlockPreference: AutoBlockPreference = AutoBlockPreference.WARN,
    val hadFalseBlock: Boolean = false
)

enum class AgeRange { UNDER_25, RANGE_25_40, RANGE_40_60, OVER_60 }
enum class Occupation { OFFICE, FREELANCE, ENTREPRENEUR, PENSIONER, STUDENT, OTHER }
enum class CallFrequency { YES, SOMETIMES, NO }
enum class BankAppCount { ZERO, ONE_TWO, THREE_PLUS }
enum class PurchaseFrequency { OFTEN, SOMETIMES, RARELY, NO }
enum class PostFrequency { OFTEN, SOMETIMES, NO }
enum class ForeignContacts { OFTEN, RARELY, NO }
enum class SpamFrequency { EVERY_DAY, SEVERAL_WEEK, RARELY, ALMOST_NONE }
enum class ScamExperience { LOST_MONEY, CAUGHT_IN_TIME, NO }
enum class AwarenessLevel { GOOD, SUPERFICIAL, NONE }
enum class ProtectionPriority { DONT_MISS, BALANCE, NO_SPAM }
enum class AnswerStrangers { ALWAYS, SOMETIMES, NEVER }
enum class AutoBlockPreference { BLOCK, WARN, DECIDE_EACH }
