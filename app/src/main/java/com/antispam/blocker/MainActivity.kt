package com.antispam.blocker

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.antispam.blocker.ui.navigation.AppNavigation
import com.antispam.blocker.ui.theme.SpamBlockerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            SpamBlockerTheme {
                val prefs = remember {
                    getSharedPreferences("spam_blocker_prefs", MODE_PRIVATE)
                }
                var onboardingNeeded by remember {
                    mutableStateOf(!prefs.getBoolean("onboarding_done", false))
                }

                AppNavigation(
                    onboardingNeeded = onboardingNeeded,
                    onOnboardingComplete = {
                        prefs.edit().putBoolean("onboarding_done", true).apply()
                        onboardingNeeded = false
                    }
                )
            }
        }
    }
}
