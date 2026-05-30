package com.antispam.blocker.data.repository

import com.antispam.blocker.data.db.dao.CallRecordDao
import com.antispam.blocker.data.db.entity.CallRecord
import com.antispam.blocker.domain.detector.Verdict
import kotlinx.coroutines.flow.Flow

class CallLogRepository(private val dao: CallRecordDao) {

    val allRecords: Flow<List<CallRecord>> = dao.getAll()

    fun recordsSince(from: Long): Flow<List<CallRecord>> = dao.getSince(from)

    fun blockedCountSince(from: Long): Flow<Int> = dao.countBlockedSince(from)

    fun warnedCountSince(from: Long): Flow<Int> = dao.countWarnedSince(from)

    fun allowedCountSince(from: Long): Flow<Int> = dao.countAllowedSince(from)

    suspend fun record(
        normalizedNumber: String?,
        originalNumber: String?,
        verdict: Verdict,
        ruleName: String? = null,
        /**
         * id строки `feature_snapshot` для ЭТОГО звонка (см. P1 фикс
         * `CallRecord.featureSnapshotId`). `null` для fast-path вердиктов.
         */
        featureSnapshotId: Long? = null,
    ) {
        dao.insert(
            CallRecord(
                normalizedNumber = normalizedNumber,
                originalNumber = originalNumber,
                verdict = verdict,
                ruleName = ruleName,
                featureSnapshotId = featureSnapshotId,
            )
        )
    }

    suspend fun deleteOlderThan(before: Long) {
        dao.deleteOlderThan(before)
    }

    /**
     * Переписать вердикт для всех записей с указанным [normalizedNumber].
     * Возвращает количество изменённых строк. Используется UI-кнопками
     * «в чёрный список» / «в белый список» в журнале — без этого вызова
     * запись остаётся со старым вердиктом и иконки не пропадают.
     */
    suspend fun updateVerdict(normalizedNumber: String, verdict: Verdict): Int {
        if (normalizedNumber.isBlank()) return 0
        return dao.updateVerdictByNumber(normalizedNumber, verdict.name)
    }
}
