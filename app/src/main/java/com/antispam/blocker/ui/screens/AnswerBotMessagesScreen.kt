package com.antispam.blocker.ui.screens

import android.media.MediaPlayer
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material.icons.outlined.ThumbDown
import androidx.compose.material.icons.outlined.ThumbUp
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.db.entity.AnswerBotMessageEntity
import com.antispam.blocker.domain.scoring.FeedbackHandler
import com.antispam.blocker.domain.scoring.RiskFactor
import com.antispam.blocker.domain.scoring.UserAction
import com.antispam.blocker.ui.theme.Amber
import com.antispam.blocker.ui.theme.BlockRed
import com.antispam.blocker.ui.theme.Ink
import com.antispam.blocker.ui.theme.TextOnAccent
import com.antispam.blocker.ui.theme.TextSecondary
import com.antispam.blocker.ui.theme.TextTertiary
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun AnswerBotMessagesScreen() {
    val app = SpamBlockerApp.instance
    val dao = remember { app.database.answerBotMessageDao() }
    val callerLookupRepo = remember { app.callerLookupRepository }
    val scope = rememberCoroutineScope()
    val blockListRepo = remember {
        com.antispam.blocker.data.repository.BlockListRepository(
            app.database.blockedNumberDao(),
            app.database.allowedNumberDao(),
            PhoneNormalizer,
        )
    }
    val feedbackHandler = remember {
        FeedbackHandler(
            feedbackStore = com.antispam.blocker.data.prefs.FeedbackLearningStore(app),
            blockListRepo = blockListRepo,
            trainingDataDao = app.database.trainingDataDao(),
        )
    }

    val messages by dao.observeAll().collectAsState(initial = emptyList())

    if (messages.isEmpty()) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Нет сообщений", style = MaterialTheme.typography.titleMedium, color = TextTertiary)
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "Расшифровки голосовых сообщений от спам-звонков появятся здесь",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextTertiary,
                )
            }
        }
        return
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        items(messages, key = { it.id }) { message ->
            AnswerBotMessageCard(
                message = message,
                callerLookupRepo = callerLookupRepo,
                onPlay = { scope.launch { playAudio(message.audioPath) } },
                onMarkSpam = {
                    scope.launch {
                        dao.markSpam(message.id, true)
                        dao.markPlayed(message.id)
                        feedbackHandler.handleFeedback(
                            message.normalizedNumber, "BLOCK",
                            UserAction.MARK_SPAM, emptyList<RiskFactor>(),
                        )
                    }
                },
                onMarkNotSpam = {
                    scope.launch {
                        dao.markSpam(message.id, false)
                        dao.markPlayed(message.id)
                        feedbackHandler.handleFeedback(
                            message.normalizedNumber, "BLOCK",
                            UserAction.NOT_SPAM, emptyList<RiskFactor>(),
                        )
                    }
                },
            )
        }
    }
}

@Composable
private fun AnswerBotMessageCard(
    message: AnswerBotMessageEntity,
    callerLookupRepo: com.antispam.blocker.domain.lookup.CallerLookupRepository,
    onPlay: () -> Unit,
    onMarkSpam: () -> Unit,
    onMarkNotSpam: () -> Unit,
) {
    val callerInfo by remember(message.normalizedNumber) {
        callerLookupRepo.observe(message.normalizedNumber)
    }.collectAsState(initial = null)

    var expanded by remember { mutableStateOf(false) }
    var playing by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    val timeStr = remember(message.timestamp) {
        SimpleDateFormat("dd MMM, HH:mm", Locale("ru")).format(Date(message.timestamp))
    }

    val durationStr = remember(message.durationMs) {
        val sec = message.durationMs / 1000
        if (sec > 0) "${sec}с" else ""
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Ink),
        shape = RoundedCornerShape(12.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = message.normalizedNumber,
                        style = MaterialTheme.typography.titleMedium,
                        color = TextOnAccent,
                    )
                    if (callerInfo != null) {
                        Text(
                            text = callerInfo!!.subtitle ?: message.normalizedNumber,
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = timeStr,
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary,
                    )
                    if (durationStr.isNotEmpty()) {
                        Text(
                            text = " · $durationStr",
                            style = MaterialTheme.typography.labelSmall,
                            color = TextTertiary,
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            if (message.transcription != null) {
                Text(
                    text = message.transcription,
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary,
                    maxLines = if (expanded) Int.MAX_VALUE else 4,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.clickable { expanded = !expanded },
                )
                if (message.transcription.length > 200) {
                    Text(
                        text = if (expanded) "Свернуть" else "Читать дальше",
                        style = MaterialTheme.typography.labelMedium,
                        color = Amber,
                        modifier = Modifier.clickable { expanded = !expanded },
                    )
                }
            } else {
                Text(
                    text = "(расшифровка недоступна)",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextTertiary,
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            if (message.spam != null) {
                Text(
                    text = if (message.spam == true) "✓ Отмечено как спам" else "✓ Отмечено как не спам",
                    style = MaterialTheme.typography.labelMedium,
                    color = if (message.spam == true) BlockRed else Amber,
                )
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    IconButton(
                        onClick = {
                            playing = !playing
                            if (playing) onPlay()
                            scope.launch {
                                withContext(Dispatchers.IO) {
                                    Thread.sleep(message.durationMs)
                                }
                                playing = false
                            }
                        },
                        modifier = Modifier.size(36.dp),
                    ) {
                        Icon(
                            imageVector = if (playing) Icons.Outlined.Stop else Icons.Outlined.PlayArrow,
                            contentDescription = "Воспроизвести",
                            tint = Amber,
                        )
                    }
                    if (playing) {
                        LinearProgressIndicator(
                            modifier = Modifier.weight(1f),
                            color = Amber,
                        )
                    } else {
                        Spacer(modifier = Modifier.weight(1f))
                    }
                    IconButton(onClick = onMarkNotSpam, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.Outlined.ThumbUp, "Не спам", tint = Amber)
                    }
                    IconButton(onClick = onMarkSpam, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.Outlined.ThumbDown, "Спам", tint = BlockRed)
                    }
                }
            }
        }
    }
}

private suspend fun playAudio(path: String) {
    withContext(Dispatchers.IO) {
        val file = File(path)
        if (!file.exists()) return@withContext
        try {
            val mp = MediaPlayer()
            mp.setDataSource(path)
            mp.prepare()
            mp.start()
            mp.setOnCompletionListener { mp.release() }
            while (mp.isPlaying) {
                Thread.sleep(100)
            }
            if (!mp.isPlaying) mp.release()
        } catch (_: Throwable) {}
    }
}
