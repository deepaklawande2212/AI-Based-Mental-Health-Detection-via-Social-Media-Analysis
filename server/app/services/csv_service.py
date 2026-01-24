"""
CSV Data Processing Service

This service handles CSV file upload, processing, and storage.
It extracts text data from CSV files for mental health analysis.

Phase 1: CSV Data Management
- File upload and validation
- CSV parsing and data extraction
- Text preprocessing for analysis
- Data storage in MongoDB
"""

import pandas as pd
import io
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from app.config.settings import settings
from app.config.database import get_csv_collection
from app.models.database_models import CSVData


class CSVService:
    """
    Service for handling CSV file operations
    
    This service provides methods to:
    1. Upload and validate CSV files
    2. Parse CSV content and extract text data
    3. Store processed data in MongoDB
    4. Retrieve stored CSV data for analysis
    """
    
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_types = settings.ALLOWED_FILE_TYPES
    
    async def process_csv_data(
        self,
        csv_content: str,
        text_column: str,
        filename: str,
        description: Optional[str] = None
    ) -> CSVData:
        """
        Process CSV content directly for analysis
        
        Args:
            csv_content: CSV content as string
            text_column: Column containing text data  
            filename: Original filename
            description: Optional description
            
        Returns:
            CSVData: Processed CSV data object
        """
        try:
            logger.info(f"📊 Processing CSV data for analysis: {filename}")
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Parse CSV content
            df = pd.read_csv(io.StringIO(csv_content))
            df.columns = df.columns.str.strip()
            
            # Get headers and validate text column
            headers = df.columns.tolist()
            if text_column not in headers:
                # Try to auto-detect if specified column doesn't exist
                text_column = self._detect_text_column(df, headers)
                if not text_column:
                    raise Exception(f"Text column not found. Available columns: {', '.join(headers)}")
            
            # Limit rows for performance
            max_rows = 5000
            if len(df) > max_rows:
                logger.warning(f"⚠️ Large CSV file, limiting to {max_rows} rows")
                df = df.head(max_rows)
            
            # Convert to list of dictionaries
            data_rows = df.fillna('').to_dict('records')
            
            # Extract text data
            text_data = []
            valid_text_rows = 0
            
            for row in data_rows:
                text = str(row.get(text_column, '')).strip()
                if len(text) > 3:  # Reduced minimum text length
                    text_data.append(text)
                    valid_text_rows += 1
            
            logger.info(f"📊 Extracted {len(text_data)} text entries from {valid_text_rows} valid rows")
            
            # Create CSVData object without database storage
            csv_data = CSVData(
                file_id=file_id,
                filename=filename,
                file_size=len(csv_content.encode('utf-8')),
                content_type="text/csv",
                headers=headers,
                data=data_rows,
                row_count=len(data_rows),
                total_rows=len(data_rows),
                valid_text_rows=valid_text_rows,
                text_column=text_column,
                processed_texts=text_data,
                description=description,
                upload_date=datetime.utcnow(),
                created_at=datetime.utcnow(),
                processed_at=datetime.utcnow(),
                processing_status="completed"
            )
            
            # Try to store in database (skip if not available)
            try:
                csv_collection = await get_csv_collection()
                await csv_collection.insert_one(csv_data.dict(exclude={"id"}))
                logger.info(f"✅ CSV data stored in database: {file_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not store CSV data in database: {str(e)}")
                logger.info("📝 Continuing without database storage")
            
            logger.success(f"✅ CSV processing completed: {len(text_data)} texts extracted")
            return csv_data
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV data: {str(e)}")
            raise Exception(f"Failed to process CSV data: {str(e)}")

    async def upload_csv_file(
        self, 
        file_content: bytes, 
        filename: str, 
        content_type: str
    ) -> CSVData:
        """
        Upload and process a CSV file
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            content_type: File content type
            
        Returns:
            CSVData: Processed CSV data object
        """
        try:
            logger.info(f"📤 Processing CSV upload: {filename}")
            
            # Validate file
            self._validate_file(file_content, content_type, filename)
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Parse CSV content
            headers, data_rows, text_column = await self._parse_csv_content(file_content)
            
            # Extract text data for analysis
            processed_texts = self._extract_text_data(data_rows, text_column)
            
            # Create CSVData object
            csv_data = CSVData(
                file_id=file_id,
                filename=filename,
                file_size=len(file_content),
                content_type=content_type,
                headers=headers,
                data=data_rows,
                row_count=len(data_rows),
                text_column=text_column,
                processed_texts=processed_texts,
                processing_status="completed",
                processed_at=datetime.utcnow()
            )
            
            # Store in database
            await self._store_csv_data(csv_data)
            
            logger.info(f"✅ Successfully processed CSV file: {filename} ({len(data_rows)} rows)")
            return csv_data
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV file {filename}: {str(e)}")
            
            # Store error information
            error_data = CSVData(
                file_id=str(uuid.uuid4()),
                filename=filename,
                file_size=len(file_content) if file_content else 0,
                content_type=content_type,
                processing_status="failed",
                error_message=str(e)
            )
            await self._store_csv_data(error_data)
            
            raise Exception(f"Failed to process CSV file: {str(e)}")
    
    def _validate_file(self, file_content: bytes, content_type: str, filename: str):
        """
        Validate uploaded file
        
        Args:
            file_content: File content
            content_type: MIME type
            filename: Original filename
            
        Raises:
            Exception: If validation fails
        """
        # Check file size
        if len(file_content) > self.max_file_size:
            raise Exception(f"File too large. Maximum size: {self.max_file_size / 1024 / 1024:.1f}MB")
        
        # Check file type
        if content_type not in self.allowed_types and not filename.lower().endswith('.csv'):
            raise Exception(f"Invalid file type. Allowed types: {', '.join(self.allowed_types)}")
        
        # Check if file is empty
        if len(file_content) == 0:
            raise Exception("File is empty")
        
        logger.info(f"✅ File validation passed: {filename}")
    
    async def _parse_csv_content(self, file_content: bytes) -> Tuple[List[str], List[Dict[str, Any]], Optional[str]]:
        """
        Parse CSV content and extract data
        
        Args:
            file_content: Raw CSV content
            
        Returns:
            Tuple: (headers, data_rows, text_column)
        """
        try:
            # Convert bytes to string
            csv_string = file_content.decode('utf-8')
            
            # Read CSV using pandas
            df = pd.read_csv(io.StringIO(csv_string))
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Get headers
            headers = df.columns.tolist()
            
            # Convert to list of dictionaries (limit rows for performance)
            max_rows = 5000  # Limit to prevent memory issues
            if len(df) > max_rows:
                logger.warning(f"⚠️ Large CSV file, limiting to {max_rows} rows")
                df = df.head(max_rows)
            
            # Convert DataFrame to list of dicts, handling NaN values
            data_rows = df.fillna('').to_dict('records')
            
            # Auto-detect text column
            text_column = self._detect_text_column(df, headers)
            
            logger.info(f"✅ CSV parsed: {len(headers)} columns, {len(data_rows)} rows")
            return headers, data_rows, text_column
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    csv_string = file_content.decode(encoding)
                    df = pd.read_csv(io.StringIO(csv_string))
                    df.columns = df.columns.str.strip()
                    headers = df.columns.tolist()
                    data_rows = df.fillna('').to_dict('records')
                    text_column = self._detect_text_column(df, headers)
                    return headers, data_rows, text_column
                except:
                    continue
            raise Exception("Unable to decode CSV file. Please ensure it's properly encoded.")
            
        except Exception as e:
            logger.error(f"❌ Error parsing CSV: {str(e)}")
            raise Exception(f"Error parsing CSV file: {str(e)}")
    
    def _detect_text_column(self, df: pd.DataFrame, headers: List[str]) -> Optional[str]:
        """
        Auto-detect the main text column for analysis
        
        Args:
            df: Pandas DataFrame
            headers: Column headers
            
        Returns:
            str or None: Detected text column name
        """
        try:
            # Common text column names
            text_keywords = [
                'text', 'content', 'message', 'comment', 'review', 'description',
                'tweet', 'post', 'feedback', 'opinion', 'statement', 'note'
            ]
            
            # Check for exact matches first
            for keyword in text_keywords:
                for header in headers:
                    if keyword.lower() in header.lower():
                        logger.info(f"🎯 Detected text column: '{header}'")
                        return header
            
            # If no exact match, find column with longest average text length
            max_avg_length = 0
            best_column = None
            
            for column in headers:
                if df[column].dtype == 'object':  # String columns
                    # Calculate average length of non-empty strings
                    avg_length = df[column].astype(str).str.len().mean()
                    if avg_length > max_avg_length and avg_length > 5:  # Reduced minimum length threshold
                        max_avg_length = avg_length
                        best_column = column
            
            # If still no column found, just pick the first string column
            if not best_column:
                for column in headers:
                    if df[column].dtype == 'object':
                        best_column = column
                        logger.info(f"🎯 Using first string column as text column: '{best_column}'")
                        break
            
            if best_column:
                logger.info(f"🎯 Auto-detected text column: '{best_column}' (avg length: {max_avg_length:.1f})")
                return best_column
            
            logger.warning("⚠️ Could not auto-detect text column")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detecting text column: {str(e)}")
            return None
    
    def _extract_text_data(self, data_rows: List[Dict[str, Any]], text_column: Optional[str]) -> List[str]:
        """
        Extract text data for analysis
        
        Args:
            data_rows: CSV data rows
            text_column: Column containing text data
            
        Returns:
            List[str]: Extracted text data
        """
        try:
            processed_texts = []
            
            if not text_column:
                # If no specific text column, concatenate all text columns
                for row in data_rows:
                    text_parts = []
                    for key, value in row.items():
                        if isinstance(value, str) and len(value.strip()) > 5:
                            text_parts.append(value.strip())
                    
                    if text_parts:
                        combined_text = " ".join(text_parts)
                        if len(combined_text) >= settings.MIN_TEXT_LENGTH:
                            processed_texts.append(combined_text)
            else:
                # Extract from specific column
                for row in data_rows:
                    text = str(row.get(text_column, '')).strip()
                    if len(text) >= settings.MIN_TEXT_LENGTH:
                        processed_texts.append(text)
            
            logger.info(f"✅ Extracted {len(processed_texts)} text entries for analysis")
            return processed_texts
            
        except Exception as e:
            logger.error(f"❌ Error extracting text data: {str(e)}")
            return []
    
    async def _store_csv_data(self, csv_data: CSVData):
        """
        Store CSV data in MongoDB
        
        Args:
            csv_data: CSVData object to store
        """
        try:
            collection = await get_csv_collection()
            
            # Convert to dict
            data_dict = csv_data.dict(exclude={"id"})
            
            # Insert into database
            result = await collection.insert_one(data_dict)
            
            logger.info(f"📝 Stored CSV data with ID: {csv_data.file_id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing CSV data: {str(e)}")
            raise
    
    async def get_csv_data(self, file_id: str) -> Optional[CSVData]:
        """
        Retrieve stored CSV data by file ID
        
        Args:
            file_id: File identifier
            
        Returns:
            CSVData or None: Stored CSV data if found
        """
        try:
            collection = await get_csv_collection()
            doc = await collection.find_one({"file_id": file_id})
            
            if doc:
                return CSVData(**doc)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving CSV data: {str(e)}")
            return None
    
    async def list_csv_files(self, limit: int = 50, skip: int = 0) -> List[CSVData]:
        """
        List stored CSV files
        
        Args:
            limit: Maximum number of files to return
            skip: Number of files to skip
            
        Returns:
            List[CSVData]: List of CSV data objects
        """
        try:
            collection = await get_csv_collection()
            
            cursor = collection.find({}).sort("upload_date", -1).skip(skip).limit(limit)
            docs = await cursor.to_list(length=limit)
            
            return [CSVData(**doc) for doc in docs]
            
        except Exception as e:
            logger.error(f"❌ Error listing CSV files: {str(e)}")
            return []
    
    async def delete_csv_data(self, file_id: str) -> bool:
        """
        Delete CSV data by file ID
        
        Args:
            file_id: File identifier
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            collection = await get_csv_collection()
            result = await collection.delete_one({"file_id": file_id})
            
            if result.deleted_count > 0:
                logger.info(f"🗑️ Deleted CSV data: {file_id}")
                return True
            else:
                logger.warning(f"⚠️ CSV data not found: {file_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error deleting CSV data: {str(e)}")
            return False
    
    def get_file_info(self, csv_data: CSVData) -> Dict[str, Any]:
        """
        Get summary information about a CSV file
        
        Args:
            csv_data: CSVData object
            
        Returns:
            Dict: File information summary
        """
        return {
            "file_id": csv_data.file_id,
            "filename": csv_data.filename,
            "file_size": csv_data.file_size,
            "row_count": csv_data.row_count,
            "columns": csv_data.headers,
            "text_column": csv_data.text_column,
            "texts_extracted": len(csv_data.processed_texts),
            "upload_date": csv_data.upload_date,
            "status": csv_data.processing_status
        }


# Global instance
csv_service = CSVService()
