from app.constant import ExceptionMessage


class InvalidFile(Exception):

    def __init__(self, file_url, message=ExceptionMessage.INVALID_FILE_EXCEPTION_MESSAGE):
        self.message = message
        self.file_url = file_url
        super().__init__(self.message)

    def __str__(self):
        return f'{self.file_url} -> {self.message}'


class FileLimitExceeded(Exception):

    def __init__(self, file_url, message=ExceptionMessage.FILE_LIMIT_EXCEEDED_EXCEPTION_MESSAGE):
        self.message = message
        self.file_url = file_url
        super().__init__(self.message)

    def __str__(self):
        return f'{self.file_url} -> {self.message}'


class HandleFileLimitExceeded(Exception):

    def __init__(self, file_url, message=ExceptionMessage.MAX_FILE_SIZE_EXCEEDED_ERROR):
        self.message = message
        self.file_url = file_url
        super().__init__(self.message)

    def __str__(self):
        return f'{self.file_url} -> {self.message}'


class FilePathNull(Exception):

    def __init__(self, message=ExceptionMessage.FILE_PATH_NULL_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class TotalPageExceeded(Exception):

    def __init__(self, page_count_threshold, message=ExceptionMessage.TOTAL_PAGE_EXCEEDED_EXCEPTION_MESSAGE):
        self.page_count_threshold = page_count_threshold
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message.format(page_count_threshold=self.page_count_threshold)


class InputQueryNull(Exception):

    def __init__(self, message=ExceptionMessage.INPUT_QUERY_NULL_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class MultipleFileUploaded(Exception):

    def __init__(self, message=ExceptionMessage.MULTIPLE_FILE_UPLOADED_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class MissingRequestBody(Exception):

    def __init__(self, message=ExceptionMessage.MISSING_REQUEST_BODY_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class InvalidRequestBody(Exception):

    def __init__(self, message=ExceptionMessage.INVALID_REQUEST_BODY_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class MissingRequiredParameter(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class InvalidFileException(Exception):

    def __init__(self, message=ExceptionMessage.INVALID_FILE_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class FileUploadLimitReached(Exception):

    def __init__(self, number_of_files, message=ExceptionMessage.FILE_UPLOAD_LIMIT_REACHED_EXCEPTION_MESSAGE):
        self.message = message
        self.number_of_files = number_of_files
        super().__init__(self.message)

    def __str__(self):
        exception_message = self.message.format(x=self.number_of_files)
        return exception_message


class FailedToDownloadFileFromURLException(Exception):

    def __init__(self, file_url, message=ExceptionMessage.FAILED_TO_DOWNLOAD_FILE_FROM_URL_EXCEPTION_MESSAGE):
        self.message = message
        self.file_url = file_url
        super().__init__(self.message)

    def __str__(self):
        return f'{self.file_url} -> {self.message}'


class NoImageFoundException(Exception):

    def __init__(self, file_url, message=ExceptionMessage.NO_IMAGE_FOUND_EXCEPTION_MESSAGE):
        self.message = message
        self.file_url = file_url
        super().__init__(self.message)

    def __str__(self):
        return f'{self.file_url} -> {self.message}'


class FolderPathNull(Exception):

    def __init__(self, message=ExceptionMessage.FOLDER_PATH_NULL_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class MissingResponseListException(Exception):

    def __init__(self, message=ExceptionMessage.MISSING_RESPONSE_LIST_EXCEPTION_MESSAGE):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'