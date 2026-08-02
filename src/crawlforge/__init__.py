"""CrawlForge package."""

from crawlforge.advanced import AdvancedCrawler
from crawlforge.chunking import ChunkingConfig, TextChunker
from crawlforge.concurrency import SemaphoreManager, SemaphoreStats
from crawlforge.config import CrawlerConfig, LoggingConfig, ReportConfig, StorageConfig
from crawlforge.content import ContentProcessor
from crawlforge.context_engine import ContextEngine, EmptyCrawlError
from crawlforge.context_index import FTS5UnavailableError, SQLiteContextIndex
from crawlforge.context_models import (
    ContextResult,
    DocumentBlock,
    HeuristicTokenEstimator,
    IndexInfo,
    IndexingResult,
    IndexSessionSummary,
    SearchHit,
    SourceDocument,
    SourceReference,
    TextChunk,
    TokenEstimator,
)
from crawlforge.crawler import AsyncCrawler, CrawledPage, CrawlStats, PageHandler
from crawlforge.errors import NetworkError, ParseError, PermanentError, TransientError
from crawlforge.hybrid import HybridRetriever, ReciprocalRankFusion
from crawlforge.hybrid_models import (
    FusedCandidate,
    HybridContextResult,
    HybridIndexReadiness,
    HybridRetrievalError,
    HybridSearchConfig,
    HybridSearchHit,
    HybridSearchMetrics,
    HybridSearchResult,
    RankContribution,
    RankedCandidate,
    RankedCandidateList,
    RankFusionStrategy,
)
from crawlforge.network_policy import URLNetworkPolicy, URLPolicyError
from crawlforge.parser import HTMLParser, ParsedPage
from crawlforge.politeness import RateLimiter, RobotsData, RobotsParser
from crawlforge.queue import CrawlerQueue, QueueStats
from crawlforge.retry import ErrorRecord, RetryStats, RetryStrategy
from crawlforge.semantic import SemanticContextEngine
from crawlforge.semantic_models import (
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL_ID,
    DEFAULT_SEMANTIC_MODEL_REVISION,
    DOCUMENT_FORMAT_VERSION,
    QUERY_FORMAT_VERSION,
    EmbeddingInputStatistics,
    EmbeddingModelInfo,
    EmbeddingProvider,
    EmbeddingRuntimeInfo,
    EmbeddingVector,
    SemanticContextResult,
    SemanticDependencyError,
    SemanticIndexIncompatibleError,
    SemanticIndexInfo,
    SemanticIndexingResult,
    SemanticIndexNotReadyError,
    SemanticSearchHit,
    SemanticSearchResult,
)
from crawlforge.semantic_provider import SentenceTransformerEmbeddingProvider
from crawlforge.sitemap import SitemapParser
from crawlforge.statistics import CrawlerStats, CrawlerStatsSnapshot, DomainCount
from crawlforge.storage import (
    CrawlData,
    CSVStorage,
    DataStorage,
    JSONStorage,
    SQLiteStorage,
)

__version__ = "0.1.0"

__all__ = [
    "AdvancedCrawler",
    "AsyncCrawler",
    "CSVStorage",
    "ChunkingConfig",
    "ContentProcessor",
    "ContextEngine",
    "ContextResult",
    "CrawledPage",
    "CrawlerQueue",
    "CrawlerConfig",
    "CrawlerStats",
    "CrawlerStatsSnapshot",
    "CrawlData",
    "CrawlStats",
    "DataStorage",
    "DEFAULT_SEMANTIC_DIMENSION",
    "DEFAULT_SEMANTIC_MODEL_ID",
    "DEFAULT_SEMANTIC_MODEL_REVISION",
    "DOCUMENT_FORMAT_VERSION",
    "DocumentBlock",
    "DomainCount",
    "EmptyCrawlError",
    "EmbeddingInputStatistics",
    "EmbeddingModelInfo",
    "EmbeddingProvider",
    "EmbeddingRuntimeInfo",
    "EmbeddingVector",
    "ErrorRecord",
    "FTS5UnavailableError",
    "FusedCandidate",
    "HTMLParser",
    "HeuristicTokenEstimator",
    "HybridContextResult",
    "HybridIndexReadiness",
    "HybridRetrievalError",
    "HybridRetriever",
    "HybridSearchConfig",
    "HybridSearchHit",
    "HybridSearchMetrics",
    "HybridSearchResult",
    "IndexInfo",
    "IndexSessionSummary",
    "IndexingResult",
    "JSONStorage",
    "LoggingConfig",
    "NetworkError",
    "ParseError",
    "ParsedPage",
    "PermanentError",
    "PageHandler",
    "QueueStats",
    "QUERY_FORMAT_VERSION",
    "RateLimiter",
    "RankContribution",
    "RankedCandidate",
    "RankedCandidateList",
    "RankFusionStrategy",
    "ReciprocalRankFusion",
    "RetryStats",
    "RetryStrategy",
    "ReportConfig",
    "RobotsData",
    "RobotsParser",
    "SemaphoreManager",
    "SemaphoreStats",
    "SemanticContextEngine",
    "SemanticContextResult",
    "SemanticDependencyError",
    "SemanticIndexIncompatibleError",
    "SemanticIndexInfo",
    "SemanticIndexingResult",
    "SemanticIndexNotReadyError",
    "SemanticSearchHit",
    "SemanticSearchResult",
    "SentenceTransformerEmbeddingProvider",
    "SQLiteStorage",
    "SQLiteContextIndex",
    "SearchHit",
    "SitemapParser",
    "SourceDocument",
    "SourceReference",
    "StorageConfig",
    "TextChunk",
    "TextChunker",
    "TokenEstimator",
    "TransientError",
    "URLNetworkPolicy",
    "URLPolicyError",
    "__version__",
]
