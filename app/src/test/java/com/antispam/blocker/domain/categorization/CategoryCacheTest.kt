package com.antispam.blocker.domain.categorization

import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Unit-тесты для [CategoryCache].
 *
 * Покрывают три аспекта:
 *  - **Capacity bound** — после прохода ≥ N различных ключей `size()` ровно N до
 *    лимита и ровно `capacity` после.
 *  - **LRU eviction** — после переполнения вытесняется самая давно использованная
 *    запись; `get` обновляет access-recency.
 *  - **Synchronization basics** — конкурентные `put`/`get` из нескольких потоков
 *    не теряют записи и не падают на `ConcurrentModificationException`.
 *
 * Полное property-based покрытие LRU-инвариантов (Property 6 из design.md, Req 7.6)
 * лежит в `CategoryCacheIdempotencyTest` и тестирует кэш через
 * `TFLiteAppCategoryClassifier` с моком `Interpreter`. Эти тесты — точечный
 * sanity-check на сам класс [CategoryCache] без ML-зависимостей.
 */
@RunWith(JUnit4::class)
class CategoryCacheTest {

    // ── Construction validation ───────────────────────────────────────────

    @Test
    fun `constructor rejects capacity below 1`() {
        assertFailsWith<IllegalArgumentException> { CategoryCache(0) }
        assertFailsWith<IllegalArgumentException> { CategoryCache(-1) }
        assertFailsWith<IllegalArgumentException> { CategoryCache(Int.MIN_VALUE) }
    }

    @Test
    fun `constructor accepts capacity of exactly 1`() {
        val cache = CategoryCache(1)
        assertEquals(0, cache.size())
        cache.put("ru.sberbankmobile", AppCategory.BANK)
        assertEquals(1, cache.size())
    }

    // ── Basic semantics ───────────────────────────────────────────────────

    @Test
    fun `get returns null for missing key`() {
        val cache = CategoryCache(8)
        assertNull(cache.get("never.put"))
    }

    @Test
    fun `put then get returns stored category`() {
        val cache = CategoryCache(8)
        cache.put("ru.sberbankmobile", AppCategory.BANK)
        assertEquals(AppCategory.BANK, cache.get("ru.sberbankmobile"))
    }

    @Test
    fun `put overwrites existing value for same key`() {
        val cache = CategoryCache(8)
        cache.put("com.example.app", AppCategory.OTHER)
        cache.put("com.example.app", AppCategory.MARKETPLACE)
        assertEquals(AppCategory.MARKETPLACE, cache.get("com.example.app"))
        assertEquals(1, cache.size())
    }

    @Test
    fun `clear removes all entries and resets size`() {
        val cache = CategoryCache(8)
        cache.put("a", AppCategory.BANK)
        cache.put("b", AppCategory.MESSENGER)
        cache.put("c", AppCategory.EMAIL)
        cache.clear()
        assertEquals(0, cache.size())
        assertNull(cache.get("a"))
        assertNull(cache.get("b"))
        assertNull(cache.get("c"))
    }

    // ── Capacity bound ────────────────────────────────────────────────────

    @Test
    fun `size never exceeds capacity even after many puts`() {
        val capacity = 4
        val cache = CategoryCache(capacity)

        for (i in 1..16) {
            cache.put("pkg.$i", AppCategory.OTHER)
            assertTrue(
                cache.size() <= capacity,
                "size=${cache.size()} exceeded capacity=$capacity at i=$i",
            )
        }
        assertEquals(capacity, cache.size())
    }

    @Test
    fun `size grows linearly until capacity is reached`() {
        val capacity = 5
        val cache = CategoryCache(capacity)
        for (i in 1..capacity) {
            cache.put("pkg.$i", AppCategory.OTHER)
            assertEquals(i, cache.size(), "after put #$i")
        }
        // следующий put не увеличивает size
        cache.put("pkg.${capacity + 1}", AppCategory.OTHER)
        assertEquals(capacity, cache.size())
    }

    // ── LRU eviction ──────────────────────────────────────────────────────

    @Test
    fun `eldest entry is evicted when capacity is exceeded`() {
        val cache = CategoryCache(3)
        cache.put("p1", AppCategory.BANK)
        cache.put("p2", AppCategory.MARKETPLACE)
        cache.put("p3", AppCategory.MESSENGER)
        // p1 ещё в кэше
        assertEquals(AppCategory.BANK, cache.get("p1"))
        // но `get(p1)` теперь сделал p1 most-recently-used; eldest = p2.
        cache.put("p4", AppCategory.EMAIL)

        assertNull(cache.get("p2"))
        assertEquals(AppCategory.BANK, cache.get("p1"))
        assertEquals(AppCategory.MESSENGER, cache.get("p3"))
        assertEquals(AppCategory.EMAIL, cache.get("p4"))
        assertEquals(3, cache.size())
    }

    @Test
    fun `lru eviction with 500 capacity matches design contract`() {
        // Эмулирует Property 6 из design.md (без ML-зависимостей):
        // последовательность p_1..p_500, потом p_501 — выселит p_1.
        val capacity = 500
        val cache = CategoryCache(capacity)
        for (i in 1..capacity) {
            cache.put("p_$i", AppCategory.OTHER)
        }
        assertEquals(capacity, cache.size())
        cache.put("p_501", AppCategory.OTHER)

        assertNull(cache.get("p_1"), "eldest p_1 must be evicted")
        for (i in 2..501) {
            assertEquals(
                AppCategory.OTHER,
                cache.get("p_$i"),
                "p_$i must remain after p_501 inserted",
            )
        }
        assertEquals(capacity, cache.size())
    }

    @Test
    fun `put refreshes access recency so subsequent eviction skips that key`() {
        val cache = CategoryCache(3)
        cache.put("p1", AppCategory.BANK)
        cache.put("p2", AppCategory.MARKETPLACE)
        cache.put("p3", AppCategory.MESSENGER)
        // re-put p1: теперь recency-порядок (от eldest к newest) = [p2, p3, p1]
        cache.put("p1", AppCategory.BANK)
        cache.put("p4", AppCategory.EMAIL)

        assertNull(cache.get("p2"), "p2 was eldest after p1 was refreshed")
        assertEquals(AppCategory.BANK, cache.get("p1"))
        assertEquals(AppCategory.MESSENGER, cache.get("p3"))
        assertEquals(AppCategory.EMAIL, cache.get("p4"))
    }

    // ── Synchronization basics ────────────────────────────────────────────

    @Test
    fun `concurrent puts from multiple threads do not crash and respect capacity`() {
        // 8 потоков по 2_000 операций — 16_000 put-ов в кэш ёмкостью 500.
        // Цель не «все ключи остались» (LRU их выселит), а «никаких
        // ConcurrentModificationException, никаких потерянных мониторов,
        // финальный size == capacity».
        val capacity = 500
        val cache = CategoryCache(capacity)
        val threads = 8
        val opsPerThread = 2_000
        val pool = Executors.newFixedThreadPool(threads)
        val ready = CountDownLatch(threads)
        val start = CountDownLatch(1)
        val done = CountDownLatch(threads)

        repeat(threads) { tid ->
            pool.execute {
                ready.countDown()
                start.await()
                try {
                    for (i in 0 until opsPerThread) {
                        val key = "t$tid.k${i % (capacity * 2)}"
                        cache.put(key, AppCategory.OTHER)
                        // перемежаем reads чтобы exercise access-order путь
                        cache.get(key)
                    }
                } finally {
                    done.countDown()
                }
            }
        }

        ready.await(5, TimeUnit.SECONDS)
        start.countDown()
        assertTrue(
            done.await(30, TimeUnit.SECONDS),
            "concurrent workers did not finish within 30s — possible deadlock",
        )
        pool.shutdownNow()

        assertEquals(capacity, cache.size())
    }

    @Test
    fun `clear concurrent with puts leaves cache in consistent state`() {
        // Сценарий: один поток непрерывно `put`-ит, второй периодически
        // `clear`-ит. После завершения size() ≤ capacity и любой `get`
        // возвращает null или валидную AppCategory без бросков.
        val capacity = 32
        val cache = CategoryCache(capacity)
        val pool = Executors.newFixedThreadPool(2)
        val running = java.util.concurrent.atomic.AtomicBoolean(true)
        val done = CountDownLatch(2)

        pool.execute {
            try {
                var i = 0
                while (running.get()) {
                    cache.put("k${i++ % 1024}", AppCategory.OTHER)
                }
            } finally {
                done.countDown()
            }
        }
        pool.execute {
            try {
                repeat(50) {
                    Thread.sleep(2)
                    cache.clear()
                }
            } finally {
                done.countDown()
            }
        }

        Thread.sleep(150)
        running.set(false)
        assertTrue(done.await(10, TimeUnit.SECONDS))
        pool.shutdownNow()

        assertTrue(
            cache.size() <= capacity,
            "size=${cache.size()} must remain ≤ capacity=$capacity",
        )
    }
}
