package com.antispam.blocker.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Block
import androidx.compose.material.icons.outlined.Call
import androidx.compose.material.icons.outlined.ChatBubbleOutline
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Block
import androidx.compose.material.icons.rounded.Call
import androidx.compose.material.icons.rounded.ChatBubble
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.antispam.blocker.ui.screens.*
import com.antispam.blocker.ui.theme.Amber
import com.antispam.blocker.ui.theme.Ink
import com.antispam.blocker.ui.theme.TextOnAccent
import com.antispam.blocker.ui.theme.TextTertiary

sealed class Screen(
    val route: String,
    val label: String,
    val iconActive: ImageVector,
    val iconInactive: ImageVector
) {
    data object Home : Screen("home", "Home", Icons.Rounded.Home, Icons.Outlined.Home)
    data object CallLog : Screen("log", "Журнал", Icons.Rounded.Call, Icons.Outlined.Call)
    data object AnswerBot : Screen("answerbot", "SMS", Icons.Rounded.ChatBubble, Icons.Outlined.ChatBubbleOutline)
    data object Blacklist : Screen("blacklist", "Список", Icons.Rounded.Block, Icons.Outlined.Block)
    data object ModelDebug : Screen("ai", "ИИ", Icons.Rounded.AutoAwesome, Icons.Outlined.AutoAwesome)
    data object Settings : Screen("settings", "Ещё", Icons.Rounded.Settings, Icons.Outlined.Settings)
}

@Composable
fun AppNavigation(onboardingNeeded: Boolean, onOnboardingComplete: () -> Unit) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    val screens = listOf(Screen.Home, Screen.CallLog, Screen.Blacklist, Screen.ModelDebug, Screen.Settings)

    val startDestination = if (onboardingNeeded) "onboarding" else Screen.Home.route

    Scaffold(
        containerColor = Ink,
        bottomBar = {
            if (currentDestination?.route != "onboarding" && currentDestination?.route != "questionnaire") {
                NavigationBar(
                    containerColor = Ink,
                    tonalElevation = 0.dp
                ) {
                    screens.forEach { screen ->
                        val selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true
                        NavigationBarItem(
                            icon = {
                                Icon(
                                    imageVector = if (selected) screen.iconActive else screen.iconInactive,
                                    contentDescription = screen.label
                                )
                            },
                            label = { Text(screen.label) },
                            selected = selected,
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = TextOnAccent,
                                selectedTextColor = Amber,
                                unselectedIconColor = TextTertiary,
                                unselectedTextColor = TextTertiary,
                                indicatorColor = Amber
                            ),
                            onClick = {
                                navController.navigate(screen.route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = startDestination,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("onboarding") {
                OnboardingScreen(onComplete = {
                    onOnboardingComplete()
                    navController.navigate("questionnaire") {
                        popUpTo("onboarding") { inclusive = true }
                    }
                })
            }
            composable("questionnaire") {
                QuestionnaireScreen(onComplete = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo("questionnaire") { inclusive = true }
                    }
                })
            }
            composable(Screen.Home.route) { HomeScreen() }
            composable(Screen.CallLog.route) { CallLogScreen(navController) }
            composable(Screen.AnswerBot.route) { AnswerBotMessagesScreen() }
            composable(Screen.Blacklist.route) { BlacklistScreen() }
            composable(Screen.ModelDebug.route) { ModelDebugScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
            composable(
                route = "explain/{snapshotId}",
                arguments = listOf(navArgument("snapshotId") { type = NavType.LongType }),
            ) { backStackEntry ->
                val id = backStackEntry.arguments?.getLong("snapshotId") ?: return@composable
                ExplainabilityDetailScreen(
                    snapshotId = id,
                    onClose = { navController.popBackStack() },
                )
            }
        }
    }
}
